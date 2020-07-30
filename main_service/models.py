from django.db import models
import datetime

class ModelResult(models.Model):
    uid = models.TextField()
    timestamp = models.DateTimeField(default=datetime.datetime.now(), blank=True)
    day_num = models.IntegerField(default=0)
    ema_order = models.SmallIntegerField(default=0)
    prediction_result = models.SmallIntegerField(default=-1)
    accuracy = models.FloatField(default=0)
    feature_ids = models.TextField()
    model_tag = models.BooleanField(default=False)
    user_tag = models.BooleanField(default=False)

    class Meta:
        unique_together = (('uid', 'day_num', 'ema_order', 'prediction_result'),)