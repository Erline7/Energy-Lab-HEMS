// src/main.ts
import { createApp } from 'vue'
import { createPinia } from 'pinia' // 👈 引入 Pinia
import App from './App.vue'
import router from './router'

const app = createApp(App)
const pinia = createPinia() // 👈 创建 Pinia 实例

app.use(pinia)  // 👈 注册 Pinia（建议写在 router 前面）
app.use(router)
app.mount('#app')