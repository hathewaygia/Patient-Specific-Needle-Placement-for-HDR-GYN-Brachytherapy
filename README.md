# Patient-Specific-Needle-Placement-for-HDR-GYN-Brachytherapy
Reinforcement Learning algorithm that predicts needle, tandem, and dwell times for patient specific anatomy in HDR GYN Hybrid Brachytherapy cases using Ir-192. Utilizes RTSTRUCT patient data and CT patient data

Run files in the following order (descriptions included)
cache_patient_masks: Build patient caches + manifest file
cache_anatomical_library: Precompute anatomical libraries (optional but recommended for speed)

train_agent: train/eval with that manifest file

You do not run these directly:
structure_utils.py, multi_patient_env.py, multi_patient_needle_env.py, rtplan_baseline.py.
They are called by the scripts above.


cache_patient_masks -> (structure_mask, label_mapping, spacing, origin, manifest)
cache_anatomical_library -> (anatomical_library.npy)
multi_patient_needle_env -> builds BrachyRL_TG43 (entire environment)
train_agent/PPO -> calls env.reset()/env.step() on BrachyRL_TG43



STEPS TO RUN THIS PROJECT FROM SCRATCH:

1. Set up Python env (3.10 works in this repo) and install dependencies
     run this in python terminal: "python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy scipy numba gymnasium stable-baselines3 torch SimpleITK rt-utils pydicom scikit-image pyvista nibabel trimesh matplotlib"

2. Put patient data in this layout (per patient):
  data/PtX Fx1/CT_Slices/ (CT DICOMs)
  data/PtX Fx1/STRUCT/ (RTSTRUCT, and ideally RTPLAN RP*.dcm)

3. Build cached masks + manifest (this is the first real prep step)
     run this in the python terminal: "python scripts/cache_patient_masks.py \
  --data-root data \
  --cache-root data/cache \
  --manifest-path data/patient_manifest.json"

4. Precompute anatomical needle libraries (recommended)
     run this in python terminal: "python scripts/cache_anatomical_library.py \
  --manifest data/patient_manifest.json"

5. Train the needle agent
   run this in python terminal: "python env/train_agent.py \
  --patient-manifest data/patient_manifest.json \
  --patient-split train \
  --eval-patient-split eval \
  --run-name my_run"

6. If you do not want to re-train, but want to evaluate, run this in python terminal:
     python env/train_agent.py \
  --patient-manifest data/patient_manifest.json \
  --patient-split train \
  --eval-patient-split eval \
  --run-name my_run \
  --export-only \
  --model-path runs/my_run/final_model.zip \
  --vecnorm-path runs/my_run/vecnormalize.pkl

*** IMPORTANT FAILURE MODES***
anatomical_lib needs HRCTV and Vagina labels, and training in manifest mode requires a baseline dose (from RTPLAN)






