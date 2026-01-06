class AnomalyScoreTarget:
    """
    이 클래스는 Grad-CAM의 '타겟'을 정의합니다.
    분류 모델의 클래스 점수 대신, 우리 모델의 '손실(loss)' 값을 타겟으로 사용합니다.
    """
    def __call__(self, model_output):
        # 모델의 출력은 (y_pred, losses, metrics) 형태의 튜플입니다.
        # 여기서 두 번째 요소인 losses 딕셔너리에서 'loss' 값을 가져옵니다.
        return model_output[1].get('loss')
