import logging
import wandb

# def extract_artifact_version(artifact, model_name):
#     for alias in artifact._attrs["aliases"]:
#         if alias["artifactCollectionName"] == model_name and alias["alias"][0] == "v" and alias["alias"][1:].isnumeric():
#             return int(alias["alias"][1:])

# def get_all_model_versions(entity, project, model_name):
#     api = wandb.Api(overrides={"entity": entity, "project": project})
#     latest_model = api.artifact(f"{model_name}:latest", type="model")
#     latest_version = extract_artifact_version(latest_model, model_name)
    
#     versions = []
#     for i in range(latest_version + 1):
#         try:
#             versions.append(api.artifact(f"{model_name}:v{i}", type="model"))
#         except Exception as e:
#             print(f"{model_name} version v{i} could not be retrieved! skipping...")
#             pass
    
#     print(f"retrieved {len(versions)} model versions!")
#     other_mdoel = wandb.Artifact(f"{entity}/{project}/{model_name}", type="model")
#     other_mdoel.add_file("model.ckpt")
#     other_mdoel.save()
#     return versions

# # run = wandb.init()
# # api = wandb.Api(overrides={"entity": "snagnar", "project": "model-registry"})
# # artifact = run.use_artifact('snagnar/model-registry/bnn-model:v1', type='model')
# # artifact_dir = artifact.download()
# versions = get_all_model_versions("snagnar", "model-registry", "bnnmodel")

# for version in versions:
#     print(f"v{extract_artifact_version(version, 'bnnmodel')}")

# # print("now melius")
# # versions = get_all_model_versions("hpi-deep-learning", "model-registry", "MeliusNet")

# # for version in versions:
# #     print(f"v{extract_artifact_version(version, 'MeliusNet')}")
# ...

from pathlib import Path
import random
# # with wandb.init(entity="hpi-deep-learning", project="model-registry") as run:
with wandb.init(entity="hpi-deep-learning", project="model-registry") as run:
    table_art = wandb.Artifact("model-tables", type="tables")
    # with Path("versions.csv").open("w") as v:
    #     v.write(f"{random.randint(0, 1e6)}")

    # with Path("versions.csv").open() as v:
    #     print("v:", v.readlines())
    table_art.add_file("versions.csv")
    # table_art.save()
    # table_art.wait()
    run.log_artifact(table_art)
    # synced_table_art = run.logged_artifacts()[0]
    run.link_artifact(table_art, "hpi-deep-learning/model-registry/model-version-tables")

# table_art.link("hpi-deep-learning/model-registry/model-version-tables")

# import wandb
# Initialize a W&B run to start tracking
# wandb.init()

# # Create an Model Version
# art = wandb.Artifact("model-tables", type="tables")

# art.add_file("versions.csv")

# # Log the Model Version
# wandb.log_artifact(art)

# # Link the Model Version to the Collection
# wandb.run.link_artifact(art, "model-registry/bnnmodel")