class Processor:
    """
    An interface object for processing examples.
    """

    def process(self, *args, **kwargs):
        """Process a single example. You must implement this."""
        raise NotImplementedError()
