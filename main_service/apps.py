from django.apps import AppConfig


class MainServiceConfig(AppConfig):
    name = 'main_service'

    def ready(self):
        import main_service.stress_prediction_service as prediction_service
        prediction_service.start()
