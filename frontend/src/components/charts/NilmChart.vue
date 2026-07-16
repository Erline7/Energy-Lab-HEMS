<template>
  <div ref="chartRef" class="chart-box"></div>
</template>

<script setup>
import { ref, shallowRef, onMounted, onUnmounted, watch } from 'vue';
import * as echarts from 'echarts';
import { useGlobalStore } from '../../store/global.js';
import { translateApp } from '../../utils/echarts.js';

const chartRef = ref(null);
const myChart = shallowRef(null);
const store = useGlobalStore();

const renderChart = () => {
  if (!myChart.value || !store.dashboardData?.nilm) return;

  const isZh = store.isZh;

  // 1. 数据映射与翻译
  const translatedData = store.dashboardData.nilm.map(item => ({
    value: item.value,
    name: translateApp(item.name, isZh)
  }));

  // 2. 计算当前正在运行的电器总功率 (用于显示在圆环正中心)
  const totalPower = translatedData.reduce((sum, item) => sum + item.value, 0).toFixed(0);

  const option = {
    // 🌟 核心修改 1：将 Tooltip 改为显示实时功率 (W)
    tooltip: {
      trigger: 'item',
      formatter: function (params) {
        return `
          <div style="font-weight:bold;margin-bottom:5px;">${params.name}</div>
          ${isZh ? '实时分解功率' : 'Real-time Power'}: <span style="color:#3b82f6;font-weight:bold;font-size:16px;">${params.value}</span> W<br/>
          ${isZh ? '当前负载占比' : 'Share'}: ${params.percent}%
        `;
      }
    },

    // 🌟 核心修改 2：在底部增加图例，让画面更像仪表盘
    legend: {
      bottom: '0%',
      left: 'center',
      icon: 'circle',
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { fontSize: 12, color: '#64748b' }
    },

    // 🌟 核心修改 3：在圆环正中心显示“当前总功率”
    title: {
      text: totalPower > 0 ? `${totalPower} W` : '0 W',
      subtext: isZh ? '当前总负荷' : 'Active Load',
      left: 'center',
      top: '36%', // 稍微偏上，给图例留出空间
      textStyle: { fontSize: 24, fontWeight: 'bold', color: '#0f172a' },
      subtextStyle: { fontSize: 12, color: '#64748b' }
    },

    series: [{
      type: 'pie',
      // 将普通饼图改为圆环图 (Donut Chart)
      radius: ['55%', '75%'],
      center: ['50%', '45%'], // 整体重心稍微往上提
      avoidLabelOverlap: false,
      itemStyle: {
        borderRadius: 4, // 让扇形边缘有圆角，更现代
        borderColor: '#fff',
        borderWidth: 2
      },
      label: { show: false }, // 隐藏外部的凌乱标签
      labelLine: { show: false },
      data: translatedData.length && totalPower > 0
            ? translatedData
            : [{ name: isZh ? '全屋待机中' : 'Standby', value: 0, itemStyle: { color: '#e2e8f0' } }]
    }]
  };

  myChart.value.setOption(option, { notMerge: true });
};

watch(() => store.dashboardData, renderChart, { deep: true });
watch(() => store.locale, renderChart);

const handleResize = () => myChart.value?.resize();
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
  min-height: 200px;
}
</style>