
class Config():
    def __init__(self):
        self.args = {
            'emo_channels' : 7,
            'params_channels' : 56,
            'wav2vec_path' : None,
            'dp' : 0.15,
            'out_norm' : None,
            'debug' : 0
        }

    def use_version(self, version:int):
        # change self.args
        pass

    def parse_args(self, version=None):
        if version != 0:
            self.use_version(version)
        return tuple(self.args.values())