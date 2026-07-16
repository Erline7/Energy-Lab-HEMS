<template>
  <div ref="chartRef" class="chart-box"></div>
</template>

<script setup>
import { ref, shallowRef, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { useGlobalStore } from '../../store/global.js';
import { translateApp, generateTimeLabels } from '../../utils/echarts.js';

const chartRef = ref(null);
const myChart = shallowRef(null);
const store = useGlobalStore();

const renderChart = () => {
  if (!myChart.value || !store.dashboardData?.timeline) return;

  const isZh = store.isZh;
  const timeLabels = generateTimeLabels();
  let appNames = [];
  const timelineData = [];
  const rawTimeline = store.dashboardData.timeline;

  if (Object.keys(rawTimeline).length > 0) {
    const appKeys = Object.keys(rawTimeline);
    appNames = appKeys.map(key => translateApp(key, isZh));
    appKeys.forEach((key, appIdx) => {
      rawTimeline[key].forEach((status, timeIdx) => timelineData.push([timeIdx, appIdx, status]));
    });
  }

  const option = {
    tooltip: { position: 'top', formatter: (p) => `${appNames[p.data[1]]}<br/>${timeLabels[p.data[0]]} : ${p.data[2] ? '🟢 ON' : '⚪ OFF'}` },
    grid: { top: '5%', bottom: '20%', left: '80', right: '2%' },
    xAxis: { type: 'category', data: timeLabels, axisLabel: { interval: 15 } },
    yAxis: { type: 'category', data: appNames, axisLabel: { fontSize: 11, fontWeight: 'bold' } },
    visualMap: { min: 0, max: 1, show: false, inRange: { color: ['#f5f7fa', '#27ae60'] } },
    series: [{ type: 'heatmap', data: timelineData }]
  };
  myChart.value.setOption(option, { notMerge: true });
};

watch(() => store.dashboardData, renderChart, { deep: true });
watch(() => store.locale, renderChart);

const handleResize = () => myChart.value?.resize();
onMounted(() => { myChart.value = echarts.init(chartRef.value); renderChart(); window.addEventListener('resize', handleResize); });
onUnmounted(() => { window.removeEventListener('resize', handleResize); myChart.value?.dispose(); });
</script>
<style scoped>.chart-box { width: 100%; height: 100%; min-height: 150px; }</style>