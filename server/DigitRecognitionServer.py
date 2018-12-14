from server import recognizer_pb2, recognizer_pb2_grpc
from recognizer import recognizer


class DigitRecognitionServer(recognizer_pb2_grpc.DigitRecognitionServicer):

    def RecognizeByPath(self, request, context):
        response = recognizer_pb2.DigitRecognitionResponse()

        service = recognizer.Recognizer()

        try:
            response.value = service.recognize_by_path(request.path)
        except Exception as e:
            print(e)

        return response

    def RecognizeByBytes(self, request, context):
        response = recognizer_pb2.DigitRecognitionResponse()

        service = recognizer.Recognizer()

        try:
            response.value = service.recognize_by_str(request.bytes)
        except Exception as e:
            print(e)

        return response
