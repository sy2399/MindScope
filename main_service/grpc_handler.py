import grpc
import json
import libs.et_service_pb2 as et_service_pb2
import libs.et_service_pb2_grpc as et_service_pb2_grpc
from main_service.feature_extraction import Features


class GrpcHandler:

    def __init__(self, server_ip_port, manager_id, manager_email, campaign_id):
        self.channel = grpc.insecure_channel(server_ip_port)
        self.stub = et_service_pb2_grpc.ETServiceStub(self.channel)
        self.manager_id = manager_id
        self.manager_email = manager_email
        self.campaign_id = campaign_id

    def grpc_close(self):
        self.channel.close()

    def grpc_load_user_emails(self):
        user_info = {}
        # retrieve participant emails
        request = et_service_pb2.RetrieveParticipantsRequestMessage(
            userId=self.manager_id,
            email=self.manager_email,
            campaignId=self.campaign_id
        )
        response = self.stub.retrieveParticipants(request)
        if not response.doneSuccessfully:
            return False
        for idx, email in enumerate(response.email):
            user_info[email] = {}
            user_info[email]['uid'] = response.userId[idx]
            # user_info.append((email, response.userId[idx]))

        for email, id in user_info.items():
            request = et_service_pb2.RetrieveParticipantStatisticsRequestMessage(
                userId=self.manager_id,
                email=self.manager_email,
                targetEmail=email,
                targetCampaignId=self.campaign_id
            )
            response = self.stub.retrieveParticipantStatistics(request)
            if not response.doneSuccessfully:
                return False



            user_info[email]['joinedTime'] = response.campaignJoinTimestamp

        return user_info

    def grpc_get_campaign_info(self):
        # retrieve campaign details --> data source ids
        request = et_service_pb2.RetrieveCampaignRequestMessage(
            userId=self.manager_id,
            email=self.manager_email,
            campaignId=self.campaign_id
        )
        response = self.stub.retrieveCampaign(request)
        if not response.doneSuccessfully:
            return None

        return response

    def grpc_load_user_data(self, from_ts, uid, data_sources, data_src_for_sleep_detection):
        # retrieve data of each participant
        data = {}
        for data_source_name in data_sources:
            # from_time for screen on and off must be more amount of data to detect sleep duration
            if data_source_name == data_src_for_sleep_detection:
                from_time = from_ts - 48 * 60 * 60 * 1000
            elif data_source_name == Features.LOCATIONS_MANUAL or data_source_name == Features.STRESS_LVL_THRESHOLDS:  # take all data for LOCATIONS_MANUAL and STRESS_LVL_THRESHOLDS
                from_time = 0
            else:
                from_time = from_ts

            data[data_source_name] = []
            data_available = True
            while data_available:
                grpc_req = et_service_pb2.Retrieve100DataRecordsRequestMessage(
                    userId=self.manager_id,
                    email=self.manager_email,
                    targetEmail=uid,
                    targetCampaignId=self.campaign_id,
                    targetDataSourceId=data_sources[data_source_name],
                    fromTimestamp=from_time
                )
                grpc_res = self.stub.retrieve100DataRecords(grpc_req)
                if grpc_res.doneSuccessfully:
                    for timestamp, value in zip(grpc_res.timestamp, grpc_res.value):
                        from_time = timestamp
                        data[data_source_name] += [(timestamp, value)]
                data_available = grpc_res.doneSuccessfully and grpc_res.moreDataAvailable
        return data

    def grpc_send_user_data(self, user_id, user_email, data_src, timestamp, value):
        req = et_service_pb2.SubmitDataRecordsRequestMessage(
            userId=user_id,
            email=user_email,
            campaignId=self.campaign_id
        )
        req.timestamp.extend([timestamp])
        req.dataSource.extend([data_src])
        req.accuracy.extend([1])
        req.values.extend([value])

        response = self.stub.submitDataRecords(req)
        print(response)
