def Pixel_based_OCR(file):
    # pretrained model
    translator = Translator()
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True, detect_language=True)
    all_text = []  # Extracted Text

    img = DocumentFile.from_images(file)
    result = model(img)

    word_values = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    word_values.append(word.value)
    lan = result.pages[0].language.get('value')

    if lan == 'en':
        # all words to strings
        string = ''.join(word_values)
        language = detect(string)
        if language == 'en':
            contents = word_values
        else:
            word_en = []
            for l in word_values:
                n_l = translator.translate(l)
                word_en.append(n_l)
            contents = word_en
            cs = [item for item in contents if not item.isnumeric()]
            cleaned_strings = [i for i in cs if i]
            all_text.append(cleaned_strings)
            # result.show()  # for showcase extracted markings
        cleaned_strings = [s.replace('-', ' ') for s in contents]

    else:

        import pytesseract
        from PIL import Image
        image = Image.open(file)
        # Extract text from the image using Hindi language
        text = pytesseract.image_to_string(image, lang='hin')
        cleaned_text = re.sub(r'[^\u0900-\u097F\s]', '', text)
        translated = translator.translate(cleaned_text, src='hi', dest='en')
        str_ = translated.text
        cleaned_string = str_.replace('\n', ' ')

        # Step 2: Split the string by spaces to create a list of words
        word_list = cleaned_string.split()
        cleaned_strings = [s.replace('-', ' ') for s in word_list]

    return cleaned_strings


# useful links -- https://github.com/tesseract-ocr/tessdata/blob/main/hin.traineddata
# https://codetoprosper.com/tesseract-ocr-for-windows/
