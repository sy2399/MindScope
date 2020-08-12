import grpc
import json
import datetime
import libs.et_service_pb2 as et_service_pb2
import libs.et_service_pb2_grpc as et_service_pb2_grpc

manager_email = 'mindscope.nsl@gmail.com'
manager_id = 21
campaign_id = 13
server_ip_port = '165.246.21.202:50051'

channel = grpc.insecure_channel(server_ip_port)
stub = et_service_pb2_grpc.ETServiceStub(channel)

request = et_service_pb2.RetrieveCampaign.Request(  # Kevin
    userId=manager_id,
    email=manager_email,
    campaignId=campaign_id
)
response = stub.retrieveCampaign(request)
if not response.doneSuccessfully:
    print("False")
else:
    data_sources = {}
    for data_source in json.loads(response.configJson):
        data_sources[data_source['name']] = data_source['data_source_id']

    print(data_sources)
