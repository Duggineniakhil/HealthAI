document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    
    // Gradient for Cyan curve (Pneumonia)
    let gradientCyan = ctx.createLinearGradient(0, 0, 0, 200);
    gradientCyan.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
    gradientCyan.addColorStop(1, 'rgba(0, 240, 255, 0.0)');

    // Gradient for Purple curve (Atelectasis)
    let gradientPurple = ctx.createLinearGradient(0, 0, 0, 200);
    gradientPurple.addColorStop(0, 'rgba(168, 85, 247, 0.4)');
    gradientPurple.addColorStop(1, 'rgba(168, 85, 247, 0.0)');

    const data = {
        labels: ['Jan', 'Feb', 'May', 'Apr', 'Jev'],
        datasets: [
            {
                label: 'Pneumonia',
                data: [5, 45, 80, 25, 95],
                borderColor: '#00f0ff',
                backgroundColor: gradientCyan,
                borderWidth: 2,
                tension: 0.4, // smooth Bezier curves
                fill: true,
                pointRadius: 0, // hide points unless hovered
                pointHoverRadius: 4
            },
            {
                label: 'Atelectasis',
                data: [10, 80, 20, 60, 10],
                borderColor: '#a855f7',
                backgroundColor: gradientPurple,
                borderWidth: 2,
                tension: 0.4, // smooth Bezier curves
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 4
            }
        ]
    };

    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false // We built custom HTML legend
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8ba0b8',
                        font: { size: 10 }
                    }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#8ba0b8',
                        stepSize: 25,
                        font: { size: 10 }
                    }
                }
            }
        }
    };

    new Chart(ctx, config);

    // Placeholder interaction for Review Scan
    const reviewBtn = document.querySelector('.btn-primary');
    reviewBtn.addEventListener('click', () => {
        alert("Initializing Clinical Review Interface...");
    });
});
