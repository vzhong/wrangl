class Processor:
    """
    Generic worker to process examples.
    """

    def process(self, *args, **kwargs):
        """Process a single example"""
        raise NotImplementedError()
