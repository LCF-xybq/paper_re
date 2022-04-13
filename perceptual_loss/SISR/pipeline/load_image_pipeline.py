from PIL import Image

class LoadImageFromFile:
    def __init__(self, key='gt', save_original_img=False,):
        self.key = key
        self.save_original_img = save_original_img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        filepath = str(results[f'{self.key}_path'])

        img = Image.read(filepath).convert('RGB')
        results[self.key] = img
        results[f'{self.key}_path'] = filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(key={self.key}, save_original_img={self.save_original_img})')
        return repr_str