<template>
  <div class="water-container">
    <div ref="chartRef" class="chart"></div>
    <div class="water-stats">
      <p>{{ store.isZh ? '今日累计:' : 'Today:' }} <span>{{ store.dashboardData?.water_gas?.water_today_total || 0 }}</span> L</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue';
import * as echarts from 'echarts';
import { useGlobalStore } from '../../store/global.js';

const store = useGlobalStore();
const chartRef = ref(null);
let chart = null;

const initChart = () => {
  if (!chartRef.value) return;
  chart = echarts.init(chartRef.value);
  const val = store.dashboardData?.water_gas?.water_current_rate || 0;

chart.setOption({
    series: [{
      type: 'gauge',

      // 🌟 核心修改 1：確保是完美半圓並重心下沉
      startAngle: 180,
      endAngle: 0,
      center: ['50%', '75%'],
      radius: '100%',

      min: 0,
      max: 5, // 根據你的實際流速調整最大值
      itemStyle: { color: '#3b82f6' },
      progress: { show: true, width: 14 }, // 加粗一點進度條會更好看
      pointer: { show: false },
      axisLine: { lineStyle: { width: 14, color: [[1, '#e2e8f0']] } },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: { show: false },

      // 🌟 核心修改 2：調整「L/h」標籤位置 (往下放)
      title: {
        show: true,
        offsetCenter: [0, '15%'], // 往下移到中心點下方
        fontSize: 14,
        color: '#64748b'
      },

      // 🌟 核心修改 3：調整「0.79」數值位置 (往上提)
      detail: {
        valueAnimation: true,
        formatter: '{value}',
        offsetCenter: [0, '-25%'], // 往上提到中心點上方
        fontSize: 28, // 數字稍微放大，更有重點感
        fontWeight: 'bold',
        color: '#334155'
      },

      data: [{ value: val, name: 'L/h' }]
    }]
  });
};

onMounted(initChart);
watch(() => store.dashboardData, initChart, { deep: true });
</script>

<style scoped>
.water-container { height: 100%; display: flex; flex-direction: column; }
.chart { flex: 1; min-height: 120px; }
.water-stats { text-align: center; font-size: 13px; color: #64748b; margin-top: -10px; }
.water-stats span { color: #3b82f6; font-weight: bold; font-size: 16px; }
</style>