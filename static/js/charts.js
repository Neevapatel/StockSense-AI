function renderStockChart(canvasId, dates, prices) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Stock Price (₹)',
                data: prices,
                borderColor: '#4f46e5', // Modern Indigo
                backgroundColor: 'rgba(79, 70, 229, 0.1)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: false }
            }
        }
    });
}