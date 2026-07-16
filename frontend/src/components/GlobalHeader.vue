<template>
  <header class="global-header">
    <div class="brand-section">
      <div class="logo-box">⚡</div>
      <h1>{{ store.isZh ? '能源实验室' : 'Energy Lab' }}</h1>
    </div>

    <nav class="view-switcher">
      <router-link to="/" class="nav-btn" active-class="active">📊 {{ store.isZh ? '数据大屏' : 'Dashboard' }}</router-link>
      <router-link to="/chat" class="nav-btn" active-class="active">🤖 {{ store.isZh ? 'AI 管家' : 'AI Assistant' }}</router-link>
    </nav>

    <div class="header-actions">
      <div class="date-picker-box">
        <span class="icon">📅</span>
        <input
          type="date"
          v-model="store.selectedDate"
          :min="minDate"
          :max="maxDate"
          @change="store.fetchDashboardData"
          class="global-date"
        />
      </div>

      <div class="lang-switch">
        <button @click="store.toggleLanguage('zh')" :class="{ active: store.locale === 'zh' }">中</button>
        <button @click="store.toggleLanguage('en')" :class="{ active: store.locale === 'en' }">EN</button>
      </div>
    </div>
  </header>
</template>

<script setup>
import { computed, onMounted } from 'vue';
import { useGlobalStore } from '../store/global.js';

const store = useGlobalStore();

// 页面加载时，触发拉取最新日期的请求
onMounted(() => {
  // 确保你的 global.js 里面已经写了 fetchLatestDbDate 方法
  if (store.fetchLatestDbDate) {
    store.fetchLatestDbDate();
  }
});

// 限制 1：可选的最早日期 (可以根据你的需要修改，或者也做成动态的)
const minDate = '2012-04-01';

// 限制 2：🌟 动态计算可选的最晚日期 (数据库最新数据的“明天”)
const maxDate = computed(() => {
  // 如果 store 里还没有加载出最新日期，暂时不设限制或给个兜底
  if (!store.latestDbDate) return '';

  const base = new Date(store.latestDbDate);
  base.setDate(base.getDate() + 1); // 往后推1天，作为日前预测的极限

  // 原生 date input 需要严格的 YYYY-MM-DD 格式
  return base.toISOString().split('T')[0];
});
</script>

<style scoped>
.global-header {
  height: 64px;
  background: #1e293b; /* 更深的蓝灰色，高级感 */
  color: white;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 100;
}

.brand-section { display: flex; align-items: center; gap: 12px; }
.logo-box { background: #3b82f6; width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 18px; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4); }
.brand-section h1 { font-size: 18px; margin: 0; font-weight: 600; letter-spacing: 1px; }

.view-switcher { display: flex; background: rgba(255,255,255,0.08); padding: 4px; border-radius: 10px; gap: 4px; }
.nav-btn { padding: 8px 24px; color: #cbd5e1; text-decoration: none; border-radius: 8px; font-size: 14px; font-weight: 600; transition: all 0.3s ease; }
.nav-btn:hover { background: rgba(255,255,255,0.1); color: white; }
.nav-btn.active { background: #3b82f6; color: white; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3); }

.header-actions { display: flex; align-items: center; gap: 16px; }

/* 日期选择器美化 */
.date-picker-box { display: flex; align-items: center; background: rgba(255,255,255,0.1); border-radius: 8px; padding: 4px 12px; gap: 8px; border: 1px solid rgba(255,255,255,0.15); }
.global-date { background: transparent; border: none; color: white; font-size: 14px; font-family: inherit; outline: none; cursor: pointer; color-scheme: dark; /* 让浏览器的日历图标变白 */ }

/* 语言切换美化 */
.lang-switch { display: flex; background: rgba(255,255,255,0.1); border-radius: 20px; padding: 3px; border: 1px solid rgba(255,255,255,0.15); }
.lang-switch button { border: none; background: transparent; padding: 4px 16px; border-radius: 16px; cursor: pointer; color: #cbd5e1; font-size: 13px; font-weight: 600; transition: all 0.3s; }
.lang-switch button.active { background: white; color: #1e293b; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>