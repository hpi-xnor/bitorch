import logging
import wandb

def extract_artifact_version(artifact, model_name):
    for alias in artifact._attrs["aliases"]:
        if alias["artifactCollectionName"] == model_name and alias["alias"][0] == "v" and alias["alias"][1:].isnumeric():
            return int(alias["alias"][1:])

def get_all_model_versions(entity, project, model_name):
    api = wandb.Api(overrides={"entity": entity, "project": project})
    latest_model = api.artifact(f"{model_name}:latest", type="model")
    latest_version = extract_artifact_version(latest_model, model_name)
    
    versions = []
    for i in range(latest_version + 1):
        try:
            versions.append(api.artifact(f"{model_name}:v{i}", type="model"))
        except Exception as e:
            print(f"{model_name} version v{i} could not be retrieved! skipping...")
            pass
    
    print(f"retrieved {len(versions)} model versions!")
    other_mdoel = wandb.Artifact(f"{entity}/{project}/{model_name}", type="model")
    other_mdoel.add_file("model.ckpt")
    other_mdoel.save()
    return versions

# run = wandb.init()
# api = wandb.Api(overrides={"entity": "snagnar", "project": "model-registry"})
# artifact = run.use_artifact('snagnar/model-registry/bnn-model:v1', type='model')
# artifact_dir = artifact.download()
versions = get_all_model_versions("snagnar", "model-registry", "bnnmodel")

for version in versions:
    print(f"v{extract_artifact_version(version, 'bnnmodel')}")

# print("now melius")
# versions = get_all_model_versions("hpi-deep-learning", "model-registry", "MeliusNet")

# for version in versions:
#     print(f"v{extract_artifact_version(version, 'MeliusNet')}")
...