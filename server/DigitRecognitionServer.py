from server import recognizer_pb2, recognizer_pb2_grpc
from recognizer import recognizer


class DigitRecognitionServer(recognizer_pb2_grpc.DigitRecognitionServicer):

    def Recognize(self, request, context):
        response = recognizer_pb2.DigitRecognitionResponse()

        service = recognizer.Recognizer()

        try:
            response.value = service.recognize(request.value)
        except Exception as e:
            print(e)

        return response
