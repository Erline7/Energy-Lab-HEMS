<template>
  <div class="app-container">
    <GlobalHeader />

    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import GlobalHeader from './components/GlobalHeader.vue';
</script>

<style>
/* 全局基础样式 */
html, body {
  margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden;
  background: #f4f7f9;
}
.app-container { display: flex; flex-direction: column; width: 100vw; height: 100vh; }
.main-content { flex: 1; position: relative; overflow: hidden; }
</style>