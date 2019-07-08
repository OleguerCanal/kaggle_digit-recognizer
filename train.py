from scripts.model import Model

if __name__ == "__main__":
    param_yaml_path = "params/basic_test.yaml"
    model = Model(param_yaml = param_yaml_path)
    model.train()