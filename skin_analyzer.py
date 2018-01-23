from network import client


class Extractor:
    """
    이미지에서 각 요소를 추출할 수 있는 함수를 제공하는 인터페이스.
    """

    def extract_erythema(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.ERYTHEMA, img, tv_1, tv_2, tv_3, tv_4, False)
        self.erythema = client.request(request_data)

    def extract_pore(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PORE, img, tv_1, tv_2, tv_3, tv_4, True)
        self.pore = client.request(request_data)

    def extract_pigmentation(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.PIGMENTATION, img, tv_1, tv_2, tv_3, tv_4, False)
        self.pigmentation = client.request(request_data)

    def extract_wrinkle(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):

        request_data = client.serialize_req_param_data(client.SkinFeature.WRINKLE, img, tv_1, tv_2, tv_3, tv_4, True)
        self.wrinkle = client.request(request_data)

    def extract_all(self, img,
                     tv_1=client.TrackBarValue.NORMAL,
                     tv_2=client.TrackBarValue.NORMAL,
                     tv_3=client.TrackBarValue.NORMAL,
                     tv_4=client.TrackBarValue.NORMAL):
        self.extract_erythema(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pore(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_pigmentation(img, tv_1, tv_2, tv_3, tv_4)
        self.extract_wrinkle(img, tv_1, tv_2, tv_3, tv_4)

    def getFeatureData(self):
        data = dict()
        data['erythema'] = self.erythema
        data['pore'] = self.pore
        data['pigmentation'] = self.pigmentation
        data['wrinkle'] = self.wrinkle
        return data


class Analyzer:
    """
    추출된 요소의 갯수 등을 이용하여 점수화하는 함수를 제공한다.
    """
    def analyze_erythema(self, erythema_data):
        num_erythema = erythema_data['Count']
        avg_area = erythema_data['Area']
        avg_darkness = erythema_data['Darkness']
        print('Erythema[ %d, %d, %d ]' % (num_erythema, avg_area, avg_darkness))

    def analyze_pore(self, pore_data):
        num_pore = pore_data['Count']
        print('Pore[ %d ]' % (num_pore))

    def analyze_pigmentation(self, pigmentation_data):
        num_pigmentation = pigmentation_data['Count']
        avg_area = pigmentation_data['Area']
        avg_darkness = pigmentation_data['Darkness']
        print('Pigmentation[ %d, %d, %d ]' % (num_pigmentation, avg_area, avg_darkness))

    def analyze_wrinkle(self, wrinkle_data):
        num_wrinkle = wrinkle_data['Count']
        avg_area = wrinkle_data['Area']
        avg_darkness = wrinkle_data['Darkness']
        pitch = wrinkle_data['Pitch']
        length = wrinkle_data['Length']
        print('Wrinkle[ %d, %d, %d, %d, %d ]' % (num_wrinkle, avg_area, avg_darkness, pitch, length))
