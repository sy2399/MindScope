import grpc

from libs import et_service_pb2_grpc, et_service_pb2

channel = grpc.insecure_channel('165.246.21.202:50051')
stub = et_service_pb2_grpc.ETServiceStub(channel)

user_email1 = 'nslabinha@gmail.com'
user_id1 = 1

user_email2 = 'hrgoh@nsl.inha.ac.kr'
user_id2 = 23

campaign_id = 13

times = [1594777680000, 1594792080000, 1594806480000, 1594820880000, 1595209680000, 1595224080000, 1595238480000, 1595252880000]
values = [
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 1, 'accuracy': 0.15517857142857144, 'model_tag': False, 'day_num': 6},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 1, 'accuracy': 0.8448214285714286, 'model_tag': True, 'day_num': 6},
     '2': {'feature_ids': '0-general_1 7-general_1 12-general_1 18-general_1 feature_29general_1', 'ema_order': 1, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 2, 'accuracy': 0.15517857142857144, 'model_tag': True, 'day_num': 6},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 2, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6},
     '2': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 2, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 3, 'accuracy': 0.15517857142857144, 'model_tag': False, 'day_num': 6},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 3, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6},
     '2': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 3, 'accuracy': 0.8448214285714286, 'model_tag': True, 'day_num': 6}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 4, 'accuracy': 0.15517857142857144, 'model_tag': True, 'day_num': 6},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 4, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6},
     '2': {'feature_ids': '0-general_1 7-general_1 12-general_1 18-general_1 feature_29general_1', 'ema_order': 4, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 6}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 1, 'accuracy': 0.15517857142857144, 'model_tag': False, 'day_num': 11},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 1, 'accuracy': 0.8448214285714286, 'model_tag': True, 'day_num': 11},
     '2': {'feature_ids': '0-general_1 7-general_1 12-general_1 18-general_1 feature_29general_1', 'ema_order': 1, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 11}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 2, 'accuracy': 0.15517857142857144, 'model_tag': True, 'day_num': 11},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 2, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 11},
     '2': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 2, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 11}},
    {'1': {'feature_ids': '29-low 24-low 20-low 8-low 21-low 10-high 6-low 2-low 26-low 28-low ', 'ema_order': 3, 'accuracy': 0.15517857142857144, 'model_tag': False, 'day_num': 11},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 3, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 11},
     '2': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 3, 'accuracy': 0.8448214285714286, 'model_tag': True, 'day_num': 11}},
    {'1': {'feature_ids': '0-general_1 7-general_1 12-general_1 18-general_1 feature_29general_1 ', 'ema_order': 4, 'accuracy': 0.15517857142857144, 'model_tag': False, 'day_num': 11},
     '0': {'feature_ids': '25-low 11-low 5-low 19-low 0-low 18-low 7-low 1-low 14-low 9-low 12-low 22-low 13-low 3-low ', 'ema_order': 4, 'accuracy': 0.8448214285714286, 'model_tag': True, 'day_num': 11},
     '2': {'feature_ids': '0-general_1 7-general_1 12-general_1 18-general_1 feature_29general_1', 'ema_order': 4, 'accuracy': 0.8448214285714286, 'model_tag': False, 'day_num': 11}}
]

# values = []
# times = [1593568800000, 1593583200000, 1593597600000, 1593612000000, 1593655200000, 1593669600000, 1593684000000, 1593698400000, 1593741600000, 1593756000000, 1593770400000, 1593784800000]
#
# values.append({"1": {'model_tag': True, 'feature_ids': '1-low 2-high 3-low 5-high 8-low 2-high 12-high 21-low 22-low', 'ema_order': 1, 'day_num': 1, 'accuracy': 50},
#                "2": {'model_tag': False, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'ema_order': 1, 'day_num': 1, 'accuracy': 30},
#                "3": {'model_tag': False, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'ema_order': 1, 'day_num': 1, 'accuracy': 20}})
# values.append({"1": {'model_tag': False, 'day_num': 1, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'ema_order': 2, 'accuracy': 30},
#                "2": {'model_tag': True, 'day_num': 1, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'ema_order': 2, 'accuracy': 50},
#                "3": {'model_tag': False, 'day_num': 1, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'ema_order': 2, 'accuracy': 20}})
# values.append({"1": {'ema_order': 3, 'model_tag': False, 'day_num': 1, 'accuracy': 30, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low'},
#                "2": {'ema_order': 3, 'model_tag': True, 'day_num': 1, 'accuracy': 50, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low'},
#                "3": {'ema_order': 3, 'model_tag': False, 'day_num': 1, 'accuracy': 20, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low'}})
# values.append({"1": {'model_tag': False, 'day_num': 1, 'accuracy': 30, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'ema_order': 4},
#                "2": {'model_tag': True, 'day_num': 1, 'accuracy': 50, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'ema_order': 4},
#                "3": {'model_tag': False, 'day_num': 1, 'accuracy': 20, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'ema_order': 4}})
# values.append({"1": {'ema_order': 1, 'day_num': 2, 'accuracy': 30, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'model_tag': False},
#                "2": {'ema_order': 1, 'day_num': 2, 'accuracy': 50, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'model_tag': True},
#                "3": {'ema_order': 1, 'day_num': 2, 'accuracy': 20, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'model_tag': False}})
# values.append({"1": {'accuracy': 30, 'model_tag': False, 'ema_order': 2, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'day_num': 2},
#                "2": {'accuracy': 50, 'model_tag': True, 'ema_order': 2, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'day_num': 2},
#                "3": {'accuracy': 20, 'model_tag': False, 'ema_order': 2, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'day_num': 2}})
# values.append({"1": {'model_tag': False, 'ema_order': 3, 'accuracy': 30, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'day_num': 2},
#                "2": {'model_tag': True, 'ema_order': 3, 'accuracy': 50, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'day_num': 2},
#                "3": {'model_tag': False, 'ema_order': 3, 'accuracy': 20, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'day_num': 2}})
# values.append({"1": {'accuracy': 30, 'model_tag': False, 'ema_order': 4, 'day_num': 2, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low'},
#                "2": {'accuracy': 50, 'model_tag': True, 'ema_order': 4, 'day_num': 2, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low'},
#                "3": {'accuracy': 20, 'model_tag': False, 'ema_order': 4, 'day_num': 2, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low'}})
# values.append({"1": {'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'day_num': 3, 'ema_order': 1, 'accuracy': 30, 'model_tag': False},
#                "2": {'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'day_num': 3, 'ema_order': 1, 'accuracy': 50, 'model_tag': True},
#                "3": {'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'day_num': 3, 'ema_order': 1, 'accuracy': 20, 'model_tag': False}})
# values.append({"1": {'model_tag': False, 'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'accuracy': 30, 'day_num': 3, 'ema_order': 2},
#                "2": {'model_tag': True, 'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'accuracy': 50, 'day_num': 3, 'ema_order': 2},
#                "3": {'model_tag': False, 'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'accuracy': 20, 'day_num': 3, 'ema_order': 2}})
# values.append({"1": {'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'ema_order': 3, 'accuracy': 30, 'model_tag': False, 'day_num': 3},
#                "2": {'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'ema_order': 3, 'accuracy': 50, 'model_tag': True, 'day_num': 3},
#                "3": {'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'ema_order': 3, 'accuracy': 20, 'model_tag': False, 'day_num': 3}})
# values.append({"1": {'feature_ids': '1-low 2-low 3-high 5-high 8-low 2-high 12-high 21-low 22-low', 'model_tag': False, 'accuracy': 30, 'day_num': 3, 'ema_order': 4},
#                "2": {'feature_ids': '1-low 2-high 3-low 4-high 8-low 7-high 10-high 21-low', 'model_tag': True, 'accuracy': 50, 'day_num': 3, 'ema_order': 4},
#                "3": {'feature_ids': '1-low 2-high 3-low 5-high 6-low 7-high 11-high 21-low', 'model_tag': False, 'accuracy': 20, 'day_num': 3, 'ema_order': 4}})

# times = [
#     1594980263000,
#     1594994423000,
#     1595052030000
# ]
#
# values = [
#     "1594980263000 3 3 3 4 4 3",
#     "1594994423000 4 2 3 4 2 3",
#     "1595052030000 2 1 2 3 4 1"
# ]

print("User 1")
for index, value in enumerate(values):
    req = et_service_pb2.SubmitDataRecordsRequestMessage(
        userId=user_id1,
        email=user_email1,
        campaignId=campaign_id
    )
    req.dataSource.extend([56])
    req.timestamp.extend([times[index]])
    req.accuracy.extend([1])
    req.values.extend([str(values[index])])
    res = stub.submitDataRecords(req)

    print(res)
    if res.doneSuccessfully:
        print('Success')
    else:
        print('failed to submit')

print("User 2")
for index, value in enumerate(values):
    req = et_service_pb2.SubmitDataRecordsRequestMessage(
        userId=user_id2,
        email=user_email2,
        campaignId=campaign_id
    )
    req.dataSource.extend([56])
    req.timestamp.extend([times[index]])
    req.accuracy.extend([1])
    req.values.extend([str(values[index])])
    res = stub.submitDataRecords(req)

    print(res)
    if res.doneSuccessfully:
        print('Success')
    else:
        print('failed to submit')
