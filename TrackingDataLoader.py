class TrackingDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.current_index = 0  # Tracks the current index
        self.first = True

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
            if self.first:
                self.first = False
            else:
                self.current_index += 1
            # print(self.current_index)
            return batch
        except StopIteration:
            # Reset the iterator and index when exhausted
            self.first = True
            self.iterator = iter(self.dataloader)
            self.current_index = 0
            # print("reseting")
            raise StopIteration

    def reset_to_position(self, index):
        """Resets the iterator to a specific position."""
        self.iterator = iter(self.dataloader)
        self.current_index = 0
        for _ in range(index):
            next(self.iterator)
            if self.first:
                self.first = False
            else:
                self.current_index += 1
            # print(self.current_index)

    def get_current_index(self):
        return self.current_index
