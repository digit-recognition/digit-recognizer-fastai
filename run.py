import grpc
from concurrent import futures
import time
from server import recognizer_pb2_grpc, DigitRecognitionServer

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    recognizer_pb2_grpc.add_DigitRecognitionServicer_to_server(DigitRecognitionServer.DigitRecognitionServer(), server)

    print("Starting server. Listening on port 50051")
    server.add_insecure_port('[::]:50051')
    server.start()

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)
