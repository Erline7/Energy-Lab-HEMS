<template>
  <div class="gas-info">
    <div class="data-item">
      <div class="label">🔥 {{ store.isZh ? '本月燃气估计' : 'Gas Monthly Pred.' }}</div>
      <div class="value">{{ store.dashboardData?.water_gas?.gas_monthly_pred || '--' }} <small>GJ</small></div>
    </div>
    <div class="data-item">
      <div class="label">🌡️ {{ store.isZh ? '目前温度参考' : 'Ref. Temperature' }}</div>
      <div class="value">{{ store.dashboardData?.water_gas?.current_temp || '--' }} <small>℃</small></div>
    </div>
  </div>
</template>

<script setup>
import { useGlobalStore } from '../../store/global.js';
const store = useGlobalStore();
</script>

<style scoped>
.gas-info {
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 100%;
  gap: 15px;
}
.data-item { border-left: 3px solid #f59e0b; padding-left: 12px; }
.label { font-size: 12px; color: #94a3b8; margin-bottom: 4px; }
.value { font-size: 22px; font-weight: bold; color: #334155; }
.value small { font-size: 12px; font-weight: normal; color: #64748b; }
</style>