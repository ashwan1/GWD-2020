from pytorch_lightning import Callback


class WriteGraphToTensorBoard(Callback):
    def __init__(self, data, *args, **kwargs):
        super(WriteGraphToTensorBoard, self).__init__(*args, **kwargs)
        self.data = data

    def on_train_start(self, trainer, pl_module):
        model = pl_module.model
        pl_module.logger.experiment.add_graph(model, self.data)
