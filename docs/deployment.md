# Deployment Guide

## Endpoints
- **Staging**: `https://complaints-api-staging-xyz123.a.run.app/predict` 
- **Production**: `https://complaints-api-prod-abc456.a.run.app/predict`
- **Vertex AI Console View**: https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?project=project-37461c1b-635d-4cf2-af5

## Monitoring
- **Dashboard**: https://console.cloud.google.com/monitoring
- **Logs**: https://console.cloud.google.com/logs

## Deployment Process
1. Push to `develop` → Staging deployment
2. Test in staging
3. Merge to `main` → Production deployment
