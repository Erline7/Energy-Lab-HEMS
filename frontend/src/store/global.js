import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../utils/api.js'

export const useGlobalStore = defineStore('global', () => {
  const locale = ref('zh')
  const selectedDate = ref('2014-04-01')
  const selectedModel = ref('qwen3:8b')

  const latestDbDate = ref('2014-04-01')

  const isZh = computed(() => locale.value === 'zh')

  const isFutureDate = computed(() => {
    if (!latestDbDate.value || !selectedDate.value) return false;
    return selectedDate.value > latestDbDate.value;
  })
  const hasData = computed(() => {
    const data = dashboardData.value;
    if (!data) return false;
    let dataExists = false;
    for (let key in data) {
      if (Array.isArray(data[key]) && data[key].length > 0 && key !== 'alerts') {
        dataExists = true;
        break;
      }
    }
    if (data.current_power !== undefined && data.current_power !== null) {
      dataExists = true;
    }
    return dataExists;
  })
  const dashboardData = ref({})
  const systemAlerts = ref([])

  const toggleLanguage = (lang) => {
    locale.value = lang
  }

  const fetchLatestDbDate = async () => {
    try {
      const res = await api.get('/api/latest_date')
      if (res.data && res.data.latest_date) {
        latestDbDate.value = res.data.latest_date
      }
    } catch (err) {
      console.error("获取最新数据日期失败:", err)
    }
  }

  const fetchDashboardData = async () => {
    try {
      const now = new Date()
      const hh = now.getHours().toString().padStart(2, '0')
      const mm = now.getMinutes().toString().padStart(2, '0')
      const ss = now.getSeconds().toString().padStart(2, '0')

      const fullDateTime = `${selectedDate.value} ${hh}:${mm}:${ss}`

      const res = await api.get(`/api/dashboard?datetime=${fullDateTime}`)
      dashboardData.value = res.data

      if (res.data && res.data.alerts) {
        systemAlerts.value = res.data.alerts
      } else {
        systemAlerts.value = []
      }

      return res.data
    } catch (err) {
      console.error('获取大屏数据失败:', err)
    }
  }

  return {
    locale,
    selectedDate,
    selectedModel,
    latestDbDate,
    isZh,
    isFutureDate,
    hasData,
    dashboardData,
    systemAlerts,
    toggleLanguage,
    fetchLatestDbDate,
    fetchDashboardData
  }
})