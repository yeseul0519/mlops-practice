\# End-to-End MLOps Pipeline



ML 모델의 학습과 실험 관리부터 모델 등록, API 서빙, 컨테이너화,

CI/CD, Kubernetes 배포, Prometheus/Grafana 기반 모니터링까지

End-to-End MLOps 흐름을 직접 구축한 실습 프로젝트입니다.



단순히 모델을 학습하는 데서 끝내지 않고,

모델을 재현 가능하게 관리하고 서비스로 배포한 뒤

운영 상태를 모니터링하는 전체 과정을 구현하는 것을 목표로 했습니다.



\---



\## Architecture



```text

Model Training

&#x20;    |

&#x20;    v

MLflow Tracking

&#x20;    |

&#x20;    v

MLflow Model Registry

(Champion Model)

&#x20;    |

&#x20;    v

FastAPI Inference API

&#x20;    |

&#x20;    v

Docker Image

&#x20;    |

&#x20;    +----------------------+

&#x20;    |                      |

&#x20;    v                      v

GitHub Actions         API Smoke Test

&#x20;    |

&#x20;    v

GitHub Container Registry

&#x20;    |

&#x20;    v

Kubernetes

&#x20; |-- Deployment (2 replicas)

&#x20; |-- Service

&#x20; |-- Readiness Probe

&#x20; `-- Liveness Probe

&#x20;    |

&#x20;    v

Prometheus

(Pod-level metrics)

&#x20;    |

&#x20;    v

Grafana Dashboard

