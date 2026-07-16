<template>
  <div class="dashboard-view">
    <div v-if="store.systemAlerts?.length > 0" class="alert-banner error-banner">
      <div v-for="(alert, i) in store.systemAlerts" :key="i">
        <span class="pulse-icon">🔴</span> {{ alert.msg }}
      </div>
    </div>

    <div v-if="store.dashboardData?.display_mode === 'PREDICTION'" class="alert-banner prediction-banner">
      <span class="pulse-icon">⚠️</span>
      {{ store.isZh ? '当前处于预测模式，所有指标均为 AI 模拟计算值' : 'Prediction Mode: All metrics are AI-simulated.' }}
    </div>

    <div class="dashboard-grid">

      <div class="card timeline-card">
        <div class="card-header">
          <h3>⏱ {{ store.isZh ? '核心家电运行状态轴' : 'Appliance Timeline' }}</h3>
        </div>
        <div class="chart-wrapper">
          <div v-if="store.dashboardData?.display_mode === 'PREDICTION' || store.isFutureDate" class="future-placeholder">
            <span class="icon">🔮</span>
            <p>{{ store.isZh ? '日前预测模式：暂无运行历史记录' : 'Forecast Mode: No history data yet' }}</p>
          </div>
          <div v-else-if="!store.hasData" class="future-placeholder">
            <span class="icon">🕳️</span>
            <p>{{ store.isZh ? '数据断层：当日无设备运行记录' : 'Data Gap: System offline this day' }}</p>
          </div>
          <TimelineChart v-else />
        </div>
      </div>

      <div class="card forecast-card" :class="{ 'prediction-mask': store.dashboardData?.display_mode === 'PREDICTION' }">
        <div class="card-header">
          <h3>📈 {{ store.isZh ? '24小时负载预测走势 (W)' : '24h Load Forecast (W)' }}</h3>
        </div>
        <div class="chart-wrapper">
          <ForecastChart />
        </div>
      </div>

      <div class="card nilm-card">
        <div class="card-header">
          <h3>🍩 {{ store.isZh ? '实时功率拆解' : 'Real-time Power Decomposition' }}</h3>
        </div>
        <div class="chart-wrapper">
          <div v-if="store.dashboardData?.display_mode === 'PREDICTION' || store.isFutureDate" class="future-placeholder">
            <span class="icon">🔮</span>
            <p>{{ store.isZh ? '预测模式：暂无实时功率' : 'Forecast Mode: No real-time data' }}</p>
          </div>
          <div v-else-if="!store.hasData" class="future-placeholder">
            <span class="icon">🕳️</span>
            <p>{{ store.isZh ? '暂无耗电数据' : 'No power data' }}</p>
          </div>
          <NilmChart v-else />
        </div>
      </div>

      <div class="card mini-card">
        <div class="card-header">
          <h3>🎯 {{ store.isZh ? '实时功率诊断' : 'Realtime Diagnosis' }}</h3>
        </div>
        <div class="chart-wrapper">
          <div v-if="store.dashboardData?.display_mode === 'PREDICTION' || store.isFutureDate" class="future-placeholder">
            <span class="icon">⏳</span>
            <p>{{ store.isZh ? '等待数据流入' : 'Waiting for data...' }}</p>
          </div>
          <div v-else-if="!store.hasData" class="future-placeholder">
            <span class="icon">🔌</span>
            <p>{{ store.isZh ? '设备未上线' : 'Device offline' }}</p>
          </div>
          <GaugeChart v-else />
        </div>
      </div>

      <div class="card mini-card" :class="{ 'prediction-mask': store.dashboardData?.display_mode === 'PREDICTION' }">
        <div class="card-header">
          <h3>💧 {{ store.isZh ? '实时水务监控' : 'Water Monitor' }}</h3>
        </div>
        <div class="chart-wrapper">
          <div v-if="!store.hasData && store.dashboardData?.display_mode !== 'PREDICTION'" class="future-placeholder">
            <span class="icon">💧</span>
            <p>{{ store.isZh ? '暂无水表读数' : 'No meter reading' }}</p>
          </div>
          <WaterChart v-else />
        </div>
      </div>

      <div class="card mini-card" :class="{ 'prediction-mask': store.dashboardData?.display_mode === 'PREDICTION' }">
        <div class="card-header">
          <h3>🔥 {{ store.isZh ? '环境与燃气分析' : 'Gas & Temp' }}</h3>
        </div>
        <div class="chart-wrapper">
          <div v-if="!store.hasData && store.dashboardData?.display_mode !== 'PREDICTION'" class="future-placeholder">
            <span class="icon">☁️</span>
            <p>{{ store.isZh ? '环境探头未响应' : 'Sensors no response' }}</p>
          </div>
          <GasCard v-else />
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue';
import { useGlobalStore } from '../store/global.js';

import ForecastChart from '../components/charts/ForecastChart.vue';
import TimelineChart from '../components/charts/TimelineChart.vue';
import GaugeChart from '../components/charts/GaugeChart.vue';
import NilmChart from '../components/charts/NilmChart.vue';
import WaterChart from '../components/charts/WaterChart.vue';
import GasCard from '../components/charts/GasCard.vue';

const store = useGlobalStore();

onMounted(() => {
  store.fetchLatestDbDate();
  store.fetchDashboardData();
});
</script>

<style scoped>
.dashboard-view {
  padding: 16px;
  background-color: #f1f5f9;
  height: calc(100vh - 64px);
  box-sizing: border-box;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: 200px minmax(300px, 1fr) 220px;
  gap: 16px;
  flex: 1;
  min-height: 760px;
}

.timeline-card { grid-column: 1 / -1; }
.forecast-card { grid-column: span 2; }
.nilm-card { grid-column: span 1; }
.mini-card { grid-column: span 1; }

.card {
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  height: 100%;
}

.card-header {
  padding: 12px 16px 0 16px;
}
.card-header h3 {
  margin: 0;
  font-size: 14px;
  color: #334155;
  font-weight: 600;
}

.chart-wrapper {
  flex: 1;
  padding: 8px 16px 16px 16px;
  position: relative;
  min-height: 0;
  height: 100%;
}

/* 警報橫幅基礎樣式 */
.alert-banner {
  padding: 10px 16px;
  border-radius: 8px;
  font-size: 13px;
  margin-bottom: 16px;
  flex-shrink: 0;
  border: 1px solid;
  border-left-width: 4px;
}

/* 系統錯誤 (紅條) */
.error-banner {
  background: #fef2f2;
  border-color: #fecaca;
  border-left-color: #ef4444;
  color: #b91c1c;
}

/* 預測模式警告 (黃條) */
.prediction-banner {
  background: #fffbeb;
  border-color: #fde68a;
  border-left-color: #f59e0b;
  color: #b45309;
  font-weight: 500;
}

.pulse-icon { animation: pulse 2s infinite; display: inline-block; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }

.future-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #94a3b8;
  text-align: center;
  background: #f8fafc;
  border-radius: 6px;
  border: 1px dashed #cbd5e1;
  margin: 4px;
}

.future-placeholder .icon {
  font-size: 28px;
  margin-bottom: 8px;
  opacity: 0.8;
  filter: grayscale(0.5);
}

.future-placeholder p {
  font-size: 13px;
  margin: 0;
  font-weight: 500;
}

/* 🌟 核心視覺：預測模式的半透明與角標遮罩 */
.prediction-mask {
  opacity: 0.85;
  transition: all 0.3s ease;
  position: relative;
}

.prediction-mask::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: repeating-linear-gradient(
    45deg,
    rgba(245, 158, 11, 0.03),
    rgba(245, 158, 11, 0.03) 10px,
    transparent 10px,
    transparent 20px
  );
  pointer-events: none;
  z-index: 10;
}

.prediction-mask::after {
  content: 'AI 模拟态';
  position: absolute;
  top: 10px;
  right: 16px;
  font-size: 11px;
  font-weight: 600;
  color: #d97706;
  background: #fef3c7;
  border: 1px solid #fde68a;
  padding: 2px 8px;
  border-radius: 12px;
  pointer-events: none;
  z-index: 11;
  box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
</style>