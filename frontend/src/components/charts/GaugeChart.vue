<template>
  <div ref="chartRef" class="chart-box"></div>
</template>

<script setup>
import { ref, shallowRef, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { useGlobalStore } from '../../store/global.js';

const chartRef = ref(null);
const myChart = shallowRef(null);
const store = useGlobalStore();

const renderChart = () => {
  if (!myChart.value || !store.dashboardData?.real_data) return;

  const validData = store.dashboardData.real_data.filter(v => v !== null && v !== undefined);
  const currentPower = validData.length ? validData[validData.length - 1] : 0;
  const isZh = store.isZh;

  const option = {
    series: [{
      name: isZh ? '实时诊断' : 'Diagnosis',
      type: 'gauge',

      // 🌟 核心修改：變成 180 度的半圓形
      startAngle: 180,
      endAngle: 0,

      // 🌟 核心修改：重心下沉，拉大半徑
      center: ['50%', '75%'], // 圓心往下移
      radius: '100%',         // 半徑撐滿容器寬度

      min: 0, max: 8000,
      splitNumber: 4,

      axisLine: {
        lineStyle: { width: 14, color: [[0.3, '#2ecc71'], [0.7, '#f39c12'], [1, '#e74c3c']] }
      },

      // 刻度線設定
      axisTick: { show: false }, // 隱藏小刻度讓畫面乾淨
      splitLine: {
        length: 18,
        lineStyle: { width: 2, color: '#fff' }
      },

      axisLabel: {
        distance: -10,
        fontSize: 12,
        color: '#64748b'
      },
    title: {
      show: true,
      // 向上偏移，第一个值为水平，第二个值为垂直
      // 这里的 '-30%' 是相对于半径的比例，负数代表向上
      offsetCenter: [0, '-25%'],
      fontSize: 13,
      color: '#64748b'
    },

    // 2. 控制 data 中的 value (519) 的位置
    detail: {
      show: true,
      // 向下偏移，正数代表向下
      // 把它放在半圆的底部开口处
      offsetCenter: [0, '30%'],
      formatter: '{value} W',
      fontSize: 22,
      fontWeight: 'bold',
      color: '#0f172a'
    },

    data: [{
      value: currentPower.toFixed(0),
      name: isZh ? '当前总负荷' : 'Current Load'
    }]
  }]
};

  myChart.value.setOption(option, { notMerge: true });
};

watch(() => store.dashboardData, renderChart, { deep: true });
watch(() => store.locale, renderChart);

const handleResize = () => myChart.value?.resize();
onMounted(() => { myChart.value = echarts.init(chartRef.value); renderChart(); window.addEventListener('resize', handleResize); });
onUnmounted(() => { window.removeEventListener('resize', handleResize); myChart.value?.dispose(); });
</script>

<style scoped>
.chart-box { width: 100%; height: 100%; min-height: 180px; }
</style>