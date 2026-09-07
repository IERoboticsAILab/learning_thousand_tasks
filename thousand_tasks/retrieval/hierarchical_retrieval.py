import glob
from os.path import join, basename, normpath, exists, isdir
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from thousand_tasks.core.globals import TASKS_DIR, ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.retrieval.auto_encoder import load_encoder, get_pcd_embd
from thousand_tasks.retrieval.language_based_retrieval import LanguageBasedRetrieval
from thousand_tasks.data.utils import load_demo_scene_state


class HierarchicalRetrieval:
    def __init__(self,
                 T_WC_demo: np.ndarray,
                 learned_tasks_dir=None,
                 T_WC_live: np.ndarray = None,
                 ):

        if learned_tasks_dir is None:
            learned_tasks_dir = TASKS_DIR

        non_task_folder_dirs = ['interaction_processed', 'bn_reaching_processed', 'processed']

        # Demo extrinsics are now read per demo from each folder's T_WC.npy, so
        # T_WC_demo survives only as the fallback for an unspecified live pose.
        self.T_WC_demo = T_WC_demo
        self.T_WC_live = T_WC_live if T_WC_live is not None else T_WC_demo
        # Geometry encoder is in assets/
        self.model_path = ASSETS_DIR
        self.root_dir = learned_tasks_dir
        self.tasks_folder_names = np.sort(
            [basename(normpath(task)) for task in glob.glob(join(self.root_dir, '*')) if
             (isdir(task) and basename(normpath(task)) not in non_task_folder_dirs)]).tolist()

        self.encoder = load_encoder(self.model_path)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
        self.language_based_retrieval = LanguageBasedRetrieval(learned_tasks_dir=self.root_dir, verbose=False)

        self._load_task_embeddings()

    def _load_task_embeddings(self):
        """Embed every usable demo, each under the camera pose it was recorded at.

        Demos are embedded here, at construction time, before any demo has been
        selected -- so each one's extrinsics must come from its own folder rather
        than from a single shared matrix.

        Demos with no `T_WC.npy` are dropped from `tasks_folder_names` and never
        become retrieval candidates.
        """

        self.embeddings = {}
        usable_task_folder_names = []
        skipped_no_extrinsics = []

        iterable = tqdm(self.tasks_folder_names)
        for task_folder_name in iterable:
            iterable.set_description('Loading embeddings of learned tasks...', refresh=False)

            # Checked before the cache below, deliberately: a demo predating
            # per-demo extrinsics may still carry a geometry_encoding.npy that
            # was computed under a single global T_WC. Keying the skip on the
            # cache instead would let such a demo through and retrieve it under
            # the wrong camera pose -- silently, with no error anywhere.
            T_WC_path = join(self.root_dir, task_folder_name, 'T_WC.npy')
            if not exists(T_WC_path):
                skipped_no_extrinsics.append(task_folder_name)
                continue
            T_WC_demo = np.load(T_WC_path)

            encoding_path = join(self.root_dir, task_folder_name, 'geometry_encoding.npy')
            if exists(encoding_path):
                self.embeddings[task_folder_name] = np.load(encoding_path)
            else:
                scene_state = load_demo_scene_state(task_name=task_folder_name,
                                                    load_segmap_if_exists=True,
                                                    learned_tasks_dir=self.root_dir)
                assert scene_state.segmap_was_set, f'Segmentation failed to load for task {task_folder_name}'
                scene_state.T_WC = T_WC_demo
                try:
                    with torch.no_grad():
                        embedding = get_pcd_embd(self.encoder, scene_state)
                except Exception as e:
                    print(f'Failed to encode demo {task_folder_name}: {e}')
                    continue

                np.save(encoding_path, embedding)
                self.embeddings[task_folder_name] = embedding

            usable_task_folder_names.append(task_folder_name)

        if skipped_no_extrinsics:
            print(f'Skipped {len(skipped_no_extrinsics)} demo(s) with no T_WC.npy '
                  f'(recorded before per-demo camera extrinsics): '
                  f'{", ".join(skipped_no_extrinsics)}')

        self.tasks_folder_names = usable_task_folder_names


    def get_task_embeddings(self, task_names: List[str]):
        embeddings = []
        for task_name in task_names:
            embeddings.append(self.embeddings[task_name])

        return np.array(embeddings)

    def get_most_similar_demo_name(self, scene_state: SceneState, template_task_description: str) -> str:

        candidate_tasks = self.language_based_retrieval.retrieve_relevant_tasks(template_task_description)

        # The language retriever scans the demo directory itself, so it can offer
        # demos that _load_task_embeddings dropped for having no T_WC.npy. Filter
        # before the length checks below -- otherwise the single-candidate branch
        # returns one outright and inference fails later trying to read its
        # extrinsics.
        candidate_tasks = [task for task in candidate_tasks if task in self.embeddings]

        if len(candidate_tasks) == 0:
            print(f'No demonstrations exist for skill {template_task_description}')
        elif len(candidate_tasks) == 1:
            return candidate_tasks[0]
        else:
            candidate_embeddings = self.get_task_embeddings(candidate_tasks)

            scene_state.T_WC = self.T_WC_live

            with torch.no_grad():
                test_embedding = np.expand_dims(get_pcd_embd(self.encoder, scene_state), 0)

            # Option 1: Cosine similarity

            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            similarity = test_embedding @ candidate_embeddings.T
            closest_task = candidate_tasks[np.argmax(similarity)]

            return closest_task
