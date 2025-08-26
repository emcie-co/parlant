const channel = new BroadcastChannel('active_tabs');

const tabId = Date.now() + '-' + Math.random();
console.log('broadcasting...');
channel.postMessage({ type: 'opened', timestamp: Date.now(), id: tabId});

window.addEventListener('beforeunload', () => {
    channel.postMessage({ tabId, type: 'closed' });
    sessionStorage.setItem('active_tabs', JSON.stringify([]));
    channel.close();
});

channel.onmessage = (event) => {
    const activeTabs = JSON.parse(sessionStorage.getItem('active_tabs') || '[]');
    if (event.data.type === 'opened' && event.data.id !== tabId) {
        sessionStorage.setItem('active_tabs', JSON.stringify([...activeTabs, event.data.id]));
    } else if (event.data.type === 'closed') {
        console.log('closedddd');
        const tabsData = activeTabs.filter((id: string) => id !== event.data.tabId);
        sessionStorage.setItem('active_tabs', JSON.stringify(tabsData));
    }
    console.log('Message from another tab:', event.data, event);
};

export const hasOtherOpenedTabs = () => {
    const tabsData = JSON.parse(sessionStorage.getItem('active_tabs') || '[]');
    return tabsData.length;
};