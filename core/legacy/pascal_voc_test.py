from core.legacy.pascal_voc_2 import pascal_voc


def test_getdata():
    pascal = pascal_voc('train')
    images, labels = pascal.get()

    print(images.shape, labels)
    # self.assertEqual(True, False)


if __name__ == '__main__':
    test_getdata()
