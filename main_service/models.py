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

class AppUsed(models.Model):
    uid = models.TextField()
    day_num = models.IntegerField(default=0)
    ema_order = models.IntegerField(default=0)


    Entertainment_Music = models.TextField()
    Utilities = models.TextField()
    Shopping  = models.TextField()
    Games_Comics = models.TextField()
    Others = models.TextField()
    Health_Wellness = models.TextField()
    Social_Communication = models.TextField()
    Education = models.TextField()
    Travel = models.TextField()
    Art_Photo = models.TextField()
    News_Magazine = models.TextField()
    Food_Drink = models.TextField()



    class Meta:
        unique_together = (('uid', 'day_num', 'ema_order'),)