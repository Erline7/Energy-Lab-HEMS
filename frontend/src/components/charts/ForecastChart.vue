<template>
  <div ref="chartRef" class="chart-box"></div>
</template>

<script setup>
import { ref, shallowRef, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { useGlobalStore } from '../../store/global.js';
import { generateTimeLabels } from '../../utils/echarts.js';

const chartRef = ref(null);
const myChart = shallowRef(null);
const store = useGlobalStore();

const renderChart = () => {
  // 如果图表还没挂载，或者没有数据，就不渲染
  if (!myChart.value || !store.dashboardData?.forecast) return;

  // 深拷贝数据，防止 ECharts 锁死 Vue 的 Proxy 响应式
  const data = JSON.parse(JSON.stringify(store.dashboardData));
  const isZh = store.isZh;
  const timeLabels = generateTimeLabels();

  const option = {
    tooltip: { trigger: 'axis' },
    legend: {
      data: [
        isZh ? '今日真实功率' : 'Real Power',
        isZh ? 'P90 极限' : 'P90 Upper',
        isZh ? 'P50 预期' : 'P50 Expected',
        isZh ? 'P10 保底' : 'P10 Lower'
      ],
      bottom: '2%'
    },
    grid: { top: '10%', bottom: '25%', left: '10%', right: '5%' },
    xAxis: { type: 'category', data: timeLabels, axisLabel: { interval: 15 } },
    yAxis: { type: 'value', name: isZh ? '功率 (W)' : 'Power (W)' },
    series: [
      { name: isZh ? '今日真实功率' : 'Real Power', data: data.real_data, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 3, color: '#f39c12' }, z: 10 },
      { name: isZh ? 'P90 极限' : 'P90 Upper', data: data.forecast.p90, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1, type: 'dashed', color: '#e74c3c' } },
      { name: isZh ? 'P50 预期' : 'P50 Expected', data: data.forecast.p50, type: 'line', smooth: true, lineStyle: { width: 3, color: '#3498db' }, areaStyle: { color: 'rgba(52,152,219,0.1)' } },
      { name: isZh ? 'P10 保底' : 'P10 Lower', data: data.forecast.p10, type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 1, type: 'dashed', color: '#2ecc71' } }
    ]
  };

  myChart.value.setOption(option, { notMerge: true });
};

// 监听大屏数据和语言的变化，只要变了就重新画图！
watch(() => store.dashboardData, renderChart, { deep: true });
watch(() => store.locale, renderChart);

const handleResize = () => { myChart.value?.resize(); };

onMounted(() => {
  myChart.value = echarts.init(chartRef.value);
  renderChart();
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  window.removeEventListener('resize', handleResize);
  myChart.value?.dispose();
});
</script>

<style scoped>
.chart-box {
  width: 100%;
  height: 100%;
  min-height: 250px; /* 保证图表有最小高度 */
}
</style>