class Configuration:
    def __init__(self, **kwargs):
        self.config = kwargs
        
    DUMP_ORDER = ["model_name", "use_system", "use_sample", "use_response", "number_of_shots", "use_temperature", "use_restriction", "source"]
    
    def dump(self):
        unknown_keys = set(self.config.keys()) - set(self.DUMP_ORDER)
        if unknown_keys:
            raise ValueError(f"Found keys in config that are not in DUMP_ORDER: {unknown_keys}")
        return [self.config[key] for key in self.DUMP_ORDER if key in self.config]
    
    def to_string(self):
        ordered_config = {key: self.config[key] for key in self.DUMP_ORDER if key in self.config}
        return "".join([f"{key}={value}-" for key, value in ordered_config.items()])[:-1]