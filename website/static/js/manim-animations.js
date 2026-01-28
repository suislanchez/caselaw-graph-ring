/**
 * Manim-Style Animations for LegalGPT Research Website
 * Inspired by 3Blue1Brown's mathematical animation style
 *
 * Features smooth SVG animations, mathematical transitions,
 * and educational visualizations for complex concepts.
 */

// ============================================================================
// Core Animation Utilities
// ============================================================================

class ManimAnimation {
    constructor(container, options = {}) {
        this.container = typeof container === 'string'
            ? document.querySelector(container)
            : container;
        this.options = {
            duration: options.duration || 2000,
            easing: options.easing || 'cubic-bezier(0.4, 0, 0.2, 1)',
            colors: {
                primary: '#3b82f6',
                secondary: '#10b981',
                accent: '#8b5cf6',
                warning: '#f59e0b',
                background: '#0f172a',
                text: '#e2e8f0',
                grid: 'rgba(59, 130, 246, 0.1)',
                ...options.colors
            },
            ...options
        };
        this.isPlaying = false;
        this.observers = [];
    }

    createSVG(width, height) {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('class', 'w-full h-auto');
        svg.style.background = this.options.colors.background;
        svg.style.borderRadius = '12px';
        return svg;
    }

    lerp(start, end, t) {
        return start + (end - start) * t;
    }

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    easeOutElastic(t) {
        const c4 = (2 * Math.PI) / 3;
        return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
    }

    animate(callback, duration = this.options.duration) {
        return new Promise(resolve => {
            const start = performance.now();
            const tick = (now) => {
                const elapsed = now - start;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.easeInOutCubic(progress);
                callback(eased, progress);
                if (progress < 1) {
                    requestAnimationFrame(tick);
                } else {
                    resolve();
                }
            };
            requestAnimationFrame(tick);
        });
    }

    observeVisibility(callback) {
        if (!this.container) return;
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !this.isPlaying) {
                    this.isPlaying = true;
                    callback();
                }
            });
        }, { threshold: 0.3 });
        observer.observe(this.container);
        this.observers.push(observer);
    }

    addReplayButton(svg) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'replay-button');
        group.style.cursor = 'pointer';
        group.style.opacity = '0';
        group.style.transition = 'opacity 0.3s';

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', '10');
        rect.setAttribute('y', '10');
        rect.setAttribute('width', '80');
        rect.setAttribute('height', '30');
        rect.setAttribute('rx', '6');
        rect.setAttribute('fill', 'rgba(59, 130, 246, 0.8)');

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', '50');
        text.setAttribute('y', '30');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', 'white');
        text.setAttribute('font-size', '12');
        text.setAttribute('font-family', 'system-ui');
        text.textContent = 'Replay';

        group.appendChild(rect);
        group.appendChild(text);
        svg.appendChild(group);

        group.addEventListener('click', () => {
            this.isPlaying = false;
            this.container.innerHTML = '';
            this.init();
        });

        return group;
    }

    showReplayButton(group) {
        group.style.opacity = '1';
    }
}


// ============================================================================
// 1. System Architecture Pipeline Animation
// ============================================================================

class PipelineAnimation extends ManimAnimation {
    constructor(container, options = {}) {
        super(container, {
            width: 800,
            height: 500,
            ...options
        });
    }

    init() {
        const svg = this.createSVG(800, 500);
        this.container.appendChild(svg);

        // Create definitions for gradients and arrows
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        // Arrow marker
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', '#10b981');
        marker.appendChild(polygon);
        defs.appendChild(marker);

        // Gradients for stages
        const stages = [
            { id: 'stage1Gradient', colors: ['#10b981', '#059669'] },
            { id: 'stage2Gradient', colors: ['#f59e0b', '#d97706'] },
            { id: 'stage3Gradient', colors: ['#8b5cf6', '#7c3aed'] }
        ];

        stages.forEach(stage => {
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.setAttribute('id', stage.id);
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '100%');
            gradient.setAttribute('y2', '100%');

            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', stage.colors[0]);

            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '100%');
            stop2.setAttribute('stop-color', stage.colors[1]);

            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            defs.appendChild(gradient);
        });

        // Glow filter
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'glow');
        filter.setAttribute('x', '-50%');
        filter.setAttribute('y', '-50%');
        filter.setAttribute('width', '200%');
        filter.setAttribute('height', '200%');
        const blur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        blur.setAttribute('stdDeviation', '3');
        blur.setAttribute('result', 'coloredBlur');
        const merge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
        const mergeNode1 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        mergeNode1.setAttribute('in', 'coloredBlur');
        const mergeNode2 = document.createElementNS('http://www.w3.org/2000/svg', 'feMergeNode');
        mergeNode2.setAttribute('in', 'SourceGraphic');
        merge.appendChild(mergeNode1);
        merge.appendChild(mergeNode2);
        filter.appendChild(blur);
        filter.appendChild(merge);
        defs.appendChild(filter);

        svg.appendChild(defs);

        // Title
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        title.setAttribute('x', '400');
        title.setAttribute('y', '35');
        title.setAttribute('text-anchor', 'middle');
        title.setAttribute('fill', '#e2e8f0');
        title.setAttribute('font-size', '18');
        title.setAttribute('font-weight', 'bold');
        title.setAttribute('font-family', 'system-ui');
        title.textContent = 'LegalGPT Pipeline Animation';
        title.style.opacity = '0';
        svg.appendChild(title);

        // Store elements for animation
        this.svg = svg;
        this.title = title;

        const replayBtn = this.addReplayButton(svg);

        this.observeVisibility(async () => {
            await this.playAnimation();
            this.showReplayButton(replayBtn);
        });
    }

    async playAnimation() {
        const svg = this.svg;

        // Fade in title
        await this.animate((t) => {
            this.title.style.opacity = t;
        }, 500);

        // Create and animate input box
        const inputBox = this.createBox(svg, 350, 60, 100, 50, '#3b82f6', 'INPUT CASE');
        await this.animateBoxIn(inputBox, 600);

        // Create data particle
        const particle = this.createParticle(svg, 400, 85);

        // Stage 1: Graph Retrieval
        await this.delay(300);
        const stage1 = this.createStageBox(svg, 100, 150, 600, 80, 'url(#stage1Gradient)',
            'STAGE 1: Graph Retrieval', ['Citation Graph', 'GraphSAGE', 'Hybrid Retriever']);
        await this.animateBoxIn(stage1, 800);
        await this.animateParticle(particle, 400, 85, 400, 150, 600);
        await this.animateParticle(particle, 400, 150, 400, 230, 400);

        // Stage 2: Context Assembly
        await this.delay(300);
        const stage2 = this.createStageBox(svg, 100, 260, 600, 80, 'url(#stage2Gradient)',
            'STAGE 2: Context Assembly', ['Prompt Template', '[INST]...precedents...[/INST]']);
        await this.animateBoxIn(stage2, 800);
        await this.animateParticle(particle, 400, 230, 400, 260, 400);
        await this.animateParticle(particle, 400, 260, 400, 340, 400);

        // Stage 3: LLM Prediction
        await this.delay(300);
        const stage3 = this.createStageBox(svg, 100, 370, 600, 80, 'url(#stage3Gradient)',
            'STAGE 3: LLM Prediction', ['Mistral-7B', 'QLoRA Adapters', 'Classification']);
        await this.animateBoxIn(stage3, 800);
        await this.animateParticle(particle, 400, 340, 400, 370, 400);
        await this.animateParticle(particle, 400, 370, 400, 450, 400);

        // Output
        await this.delay(300);
        const outputBox = this.createBox(svg, 300, 460, 200, 35, '#3b82f6', 'P(petitioner) | P(respondent)');
        await this.animateBoxIn(outputBox, 600);

        // Final particle burst
        await this.animateParticleBurst(particle, 400, 475);
    }

    createBox(svg, x, y, width, height, fill, text) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.style.opacity = '0';
        group.style.transform = 'scale(0.8)';
        group.style.transformOrigin = `${x + width/2}px ${y + height/2}px`;

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('rx', '8');
        rect.setAttribute('fill', fill);
        rect.setAttribute('filter', 'url(#glow)');

        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x + width/2);
        label.setAttribute('y', y + height/2 + 5);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', 'white');
        label.setAttribute('font-size', '12');
        label.setAttribute('font-family', 'system-ui');
        label.textContent = text;

        group.appendChild(rect);
        group.appendChild(label);
        svg.appendChild(group);
        return group;
    }

    createStageBox(svg, x, y, width, height, fill, title, subItems) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.style.opacity = '0';
        group.style.transform = 'translateX(-50px)';

        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('rx', '12');
        rect.setAttribute('fill', fill);
        rect.setAttribute('filter', 'url(#glow)');

        const titleText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        titleText.setAttribute('x', x + width/2);
        titleText.setAttribute('y', y + 25);
        titleText.setAttribute('text-anchor', 'middle');
        titleText.setAttribute('fill', 'white');
        titleText.setAttribute('font-size', '14');
        titleText.setAttribute('font-weight', 'bold');
        titleText.setAttribute('font-family', 'system-ui');
        titleText.textContent = title;

        group.appendChild(rect);
        group.appendChild(titleText);

        // Add sub-items
        const itemWidth = width / subItems.length;
        subItems.forEach((item, i) => {
            const subRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            subRect.setAttribute('x', x + 20 + i * (itemWidth - 10));
            subRect.setAttribute('y', y + 40);
            subRect.setAttribute('width', itemWidth - 30);
            subRect.setAttribute('height', '30');
            subRect.setAttribute('rx', '6');
            subRect.setAttribute('fill', 'rgba(255,255,255,0.2)');

            const subText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            subText.setAttribute('x', x + 20 + i * (itemWidth - 10) + (itemWidth - 30)/2);
            subText.setAttribute('y', y + 60);
            subText.setAttribute('text-anchor', 'middle');
            subText.setAttribute('fill', 'white');
            subText.setAttribute('font-size', '11');
            subText.setAttribute('font-family', 'system-ui');
            subText.textContent = item;

            group.appendChild(subRect);
            group.appendChild(subText);
        });

        svg.appendChild(group);
        return group;
    }

    createParticle(svg, x, y) {
        const particle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        particle.setAttribute('cx', x);
        particle.setAttribute('cy', y);
        particle.setAttribute('r', '8');
        particle.setAttribute('fill', '#10b981');
        particle.setAttribute('filter', 'url(#glow)');
        particle.style.opacity = '0';
        svg.appendChild(particle);
        return particle;
    }

    async animateBoxIn(element, duration) {
        await this.animate((t) => {
            element.style.opacity = t;
            element.style.transform = `scale(${0.8 + 0.2 * t}) translateX(${-50 * (1-t)}px)`;
        }, duration);
    }

    async animateParticle(particle, x1, y1, x2, y2, duration) {
        particle.style.opacity = '1';
        await this.animate((t) => {
            const x = this.lerp(x1, x2, t);
            const y = this.lerp(y1, y2, t);
            particle.setAttribute('cx', x);
            particle.setAttribute('cy', y);
            particle.setAttribute('r', 8 + Math.sin(t * Math.PI) * 4);
        }, duration);
    }

    async animateParticleBurst(particle, x, y) {
        // Create burst particles
        const svg = this.svg;
        const particles = [];
        for (let i = 0; i < 8; i++) {
            const p = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            p.setAttribute('cx', x);
            p.setAttribute('cy', y);
            p.setAttribute('r', '4');
            p.setAttribute('fill', i % 2 === 0 ? '#10b981' : '#3b82f6');
            p.setAttribute('filter', 'url(#glow)');
            svg.appendChild(p);
            particles.push({ el: p, angle: (i / 8) * Math.PI * 2 });
        }
        particle.style.opacity = '0';

        await this.animate((t) => {
            particles.forEach(p => {
                const dist = t * 40;
                const px = x + Math.cos(p.angle) * dist;
                const py = y + Math.sin(p.angle) * dist;
                p.el.setAttribute('cx', px);
                p.el.setAttribute('cy', py);
                p.el.style.opacity = 1 - t;
                p.el.setAttribute('r', 4 * (1 - t * 0.5));
            });
        }, 600);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}


// ============================================================================
// 2. GraphSAGE Message Passing Animation
// ============================================================================

class GraphSAGEAnimation extends ManimAnimation {
    constructor(container, options = {}) {
        super(container, { width: 700, height: 400, ...options });
    }

    init() {
        const svg = this.createSVG(700, 400);
        this.container.appendChild(svg);
        this.svg = svg;

        // Definitions
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        // Glow filter
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'nodeGlow');
        filter.setAttribute('x', '-100%');
        filter.setAttribute('y', '-100%');
        filter.setAttribute('width', '300%');
        filter.setAttribute('height', '300%');
        const blur = document.createElementNS('http://www.w3.org/2000/svg', 'feGaussianBlur');
        blur.setAttribute('stdDeviation', '4');
        blur.setAttribute('result', 'coloredBlur');
        const merge = document.createElementNS('http://www.w3.org/2000/svg', 'feMerge');
        merge.innerHTML = '<feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/>';
        filter.appendChild(blur);
        filter.appendChild(merge);
        defs.appendChild(filter);
        svg.appendChild(defs);

        // Title
        this.title = this.createText(350, 30, 'GraphSAGE Message Passing', 16, 'bold');
        this.title.style.opacity = '0';

        // Formula text
        this.formula = this.createText(350, 370, 'h_v = MEAN({h_u : u in N(v)})', 14, 'normal', '#10b981');
        this.formula.setAttribute('font-family', 'monospace');
        this.formula.style.opacity = '0';

        const replayBtn = this.addReplayButton(svg);

        this.observeVisibility(async () => {
            await this.playAnimation();
            this.showReplayButton(replayBtn);
        });
    }

    createText(x, y, text, size, weight, fill = '#e2e8f0') {
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('fill', fill);
        t.setAttribute('font-size', size);
        t.setAttribute('font-weight', weight);
        t.setAttribute('font-family', 'system-ui');
        t.textContent = text;
        this.svg.appendChild(t);
        return t;
    }

    createNode(x, y, r, color, label) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', color);
        circle.setAttribute('filter', 'url(#nodeGlow)');

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y + 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', 'white');
        text.setAttribute('font-size', '12');
        text.setAttribute('font-weight', 'bold');
        text.textContent = label;

        group.appendChild(circle);
        group.appendChild(text);
        group.style.opacity = '0';
        this.svg.appendChild(group);
        return { group, circle };
    }

    createEdge(x1, y1, x2, y2) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x1);
        line.setAttribute('y2', y1);
        line.setAttribute('stroke', 'rgba(59, 130, 246, 0.5)');
        line.setAttribute('stroke-width', '2');
        line.dataset.targetX = x2;
        line.dataset.targetY = y2;
        this.svg.insertBefore(line, this.svg.firstChild.nextSibling);
        return line;
    }

    async playAnimation() {
        // Fade in title
        await this.animate((t) => {
            this.title.style.opacity = t;
        }, 500);

        // Create central node (v)
        const centerNode = this.createNode(350, 200, 35, '#8b5cf6', 'v');
        await this.animateNodeIn(centerNode.group, 600);

        // Create neighbor nodes
        const neighbors = [
            { x: 200, y: 120, label: 'u1' },
            { x: 500, y: 120, label: 'u2' },
            { x: 150, y: 280, label: 'u3' },
            { x: 550, y: 280, label: 'u4' },
            { x: 350, y: 80, label: 'u5' }
        ];

        const edges = [];
        const nodes = [];

        // Create edges first (behind nodes)
        for (const n of neighbors) {
            const edge = this.createEdge(350, 200, n.x, n.y);
            edges.push(edge);
        }

        // Create neighbor nodes
        for (const n of neighbors) {
            const node = this.createNode(n.x, n.y, 25, '#3b82f6', n.label);
            nodes.push(node);
        }

        // Animate nodes appearing
        for (let i = 0; i < nodes.length; i++) {
            await this.animateNodeIn(nodes[i].group, 300);
            // Animate edge
            await this.animateEdge(edges[i], 300);
        }

        await this.delay(500);

        // Phase label
        const phase1 = this.createText(350, 340, 'Step 1: AGGREGATE neighbors', 12, 'normal', '#f59e0b');
        phase1.style.opacity = '0';
        await this.animate((t) => { phase1.style.opacity = t; }, 400);

        // Animate aggregation - nodes pulse and send info to center
        await this.animateAggregation(nodes, centerNode, edges);

        await this.delay(300);
        phase1.style.opacity = '0';
        const phase2 = this.createText(350, 340, 'Step 2: COMBINE with self-embedding', 12, 'normal', '#10b981');
        phase2.style.opacity = '0';
        await this.animate((t) => { phase2.style.opacity = t; }, 400);

        // Animate center node transformation
        await this.animateCombine(centerNode);

        await this.delay(300);
        phase2.style.opacity = '0';

        // Show formula
        await this.animate((t) => {
            this.formula.style.opacity = t;
        }, 600);
    }

    async animateNodeIn(group, duration) {
        group.style.transformOrigin = 'center';
        group.style.transform = 'scale(0)';
        group.style.opacity = '1';
        await this.animate((t) => {
            const scale = this.easeOutElastic(t);
            group.style.transform = `scale(${scale})`;
        }, duration);
    }

    async animateEdge(edge, duration) {
        const x1 = parseFloat(edge.getAttribute('x1'));
        const y1 = parseFloat(edge.getAttribute('y1'));
        const x2 = parseFloat(edge.dataset.targetX);
        const y2 = parseFloat(edge.dataset.targetY);

        await this.animate((t) => {
            edge.setAttribute('x2', this.lerp(x1, x2, t));
            edge.setAttribute('y2', this.lerp(y1, y2, t));
        }, duration);
    }

    async animateAggregation(nodes, centerNode, edges) {
        // Pulse each neighbor and create flowing particles
        for (let i = 0; i < nodes.length; i++) {
            const node = nodes[i];
            const edge = edges[i];

            // Pulse the neighbor
            node.circle.style.transition = 'fill 0.3s';
            node.circle.setAttribute('fill', '#10b981');

            // Create flowing particle
            const cx = parseFloat(node.circle.getAttribute('cx'));
            const cy = parseFloat(node.circle.getAttribute('cy'));

            const particle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            particle.setAttribute('cx', cx);
            particle.setAttribute('cy', cy);
            particle.setAttribute('r', '6');
            particle.setAttribute('fill', '#10b981');
            particle.setAttribute('filter', 'url(#nodeGlow)');
            this.svg.appendChild(particle);

            await this.animate((t) => {
                particle.setAttribute('cx', this.lerp(cx, 350, t));
                particle.setAttribute('cy', this.lerp(cy, 200, t));
                particle.style.opacity = 1 - t * 0.5;
            }, 400);

            particle.remove();
            node.circle.setAttribute('fill', '#3b82f6');

            await this.delay(100);
        }
    }

    async animateCombine(centerNode) {
        // Animate the center node growing and changing color
        await this.animate((t) => {
            const r = 35 + Math.sin(t * Math.PI) * 15;
            centerNode.circle.setAttribute('r', r);

            // Color transition from purple to green
            const r1 = Math.round(139 + (16 - 139) * t);
            const g1 = Math.round(92 + (185 - 92) * t);
            const b1 = Math.round(246 + (129 - 246) * t);
            centerNode.circle.setAttribute('fill', `rgb(${r1}, ${g1}, ${b1})`);
        }, 1000);

        // Final pulse
        await this.animate((t) => {
            const scale = 1 + Math.sin(t * Math.PI * 2) * 0.1;
            centerNode.group.style.transform = `scale(${scale})`;
        }, 500);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}


// ============================================================================
// 3. Citation Network Graph Animation
// ============================================================================

class CitationNetworkAnimation extends ManimAnimation {
    constructor(container, options = {}) {
        super(container, { width: 700, height: 450, ...options });
    }

    init() {
        const svg = this.createSVG(700, 450);
        this.container.appendChild(svg);
        this.svg = svg;

        // Defs
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        // Arrow marker
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'citationArrow');
        marker.setAttribute('markerWidth', '8');
        marker.setAttribute('markerHeight', '6');
        marker.setAttribute('refX', '8');
        marker.setAttribute('refY', '3');
        marker.setAttribute('orient', 'auto');
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', '0 0, 8 3, 0 6');
        poly.setAttribute('fill', 'rgba(59, 130, 246, 0.7)');
        marker.appendChild(poly);
        defs.appendChild(marker);

        // Glow
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'caseGlow');
        filter.innerHTML = '<feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>';
        defs.appendChild(filter);

        svg.appendChild(defs);

        // Title
        this.title = this.createText(350, 30, 'Citation Network Construction', 16, 'bold');
        this.title.style.opacity = '0';

        // Stats that will appear
        this.stats = this.createText(350, 420, '', 12, 'normal', '#10b981');
        this.stats.style.opacity = '0';

        const replayBtn = this.addReplayButton(svg);

        this.observeVisibility(async () => {
            await this.playAnimation();
            this.showReplayButton(replayBtn);
        });
    }

    createText(x, y, text, size, weight, fill = '#e2e8f0') {
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('fill', fill);
        t.setAttribute('font-size', size);
        t.setAttribute('font-weight', weight);
        t.setAttribute('font-family', 'system-ui');
        t.textContent = text;
        this.svg.appendChild(t);
        return t;
    }

    async playAnimation() {
        // Fade in title
        await this.animate((t) => {
            this.title.style.opacity = t;
        }, 500);

        // Define case nodes - arranged in a nice pattern
        const cases = [
            { x: 350, y: 200, label: 'Roe v. Wade', color: '#10b981', size: 30 },
            { x: 200, y: 150, label: 'Griswold', color: '#3b82f6', size: 22 },
            { x: 500, y: 150, label: 'Casey', color: '#3b82f6', size: 24 },
            { x: 150, y: 280, label: 'Eisenstadt', color: '#8b5cf6', size: 20 },
            { x: 280, y: 320, label: 'Carey', color: '#8b5cf6', size: 18 },
            { x: 420, y: 320, label: 'Webster', color: '#f59e0b', size: 20 },
            { x: 550, y: 280, label: 'Akron', color: '#f59e0b', size: 18 },
            { x: 350, y: 100, label: 'Dobbs', color: '#ef4444', size: 26 }
        ];

        // Citations (from -> to)
        const citations = [
            [1, 0], [2, 0], [3, 1], [4, 0], [5, 2], [6, 2],
            [7, 0], [7, 2], [3, 0], [4, 1], [5, 0], [6, 0]
        ];

        const nodeElements = [];
        const edgeElements = [];

        // Create edges first (behind nodes)
        const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.svg.insertBefore(edgesGroup, this.svg.firstChild.nextSibling);

        for (const [from, to] of citations) {
            const edge = this.createEdge(edgesGroup, cases[from], cases[to]);
            edgeElements.push(edge);
        }

        // Animate nodes appearing one by one
        let nodeCount = 0;
        let edgeCount = 0;

        for (let i = 0; i < cases.length; i++) {
            const c = cases[i];
            const node = this.createCaseNode(c.x, c.y, c.size, c.color, c.label);
            nodeElements.push(node);
            await this.animateNodeAppear(node, 300);
            nodeCount++;

            // Update stats
            this.stats.textContent = `Nodes: ${nodeCount} | Edges: ${edgeCount}`;
            await this.animate((t) => { this.stats.style.opacity = t; }, 100);

            // Animate edges connected to this node
            for (let j = 0; j < citations.length; j++) {
                if (citations[j][0] === i && citations[j][1] < i) {
                    await this.animateEdge(edgeElements[j], 200);
                    edgeCount++;
                    this.stats.textContent = `Nodes: ${nodeCount} | Edges: ${edgeCount}`;
                }
            }

            await this.delay(150);
        }

        // Animate remaining edges
        for (let j = 0; j < citations.length; j++) {
            if (edgeElements[j].style.opacity !== '1') {
                await this.animateEdge(edgeElements[j], 150);
                edgeCount++;
                this.stats.textContent = `Nodes: ${nodeCount} | Edges: ${edgeCount}`;
            }
        }

        // Final highlight animation
        await this.delay(500);
        await this.animateNetworkPulse(nodeElements);
    }

    createCaseNode(x, y, r, color, label) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.style.opacity = '0';

        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', color);
        circle.setAttribute('filter', 'url(#caseGlow)');

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x);
        text.setAttribute('y', y + r + 15);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#94a3b8');
        text.setAttribute('font-size', '10');
        text.textContent = label;

        group.appendChild(circle);
        group.appendChild(text);
        this.svg.appendChild(group);
        return { group, circle, x, y };
    }

    createEdge(parent, from, to) {
        // Calculate direction vector
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const len = Math.sqrt(dx * dx + dy * dy);
        const nx = dx / len;
        const ny = dy / len;

        // Offset start and end by node radius
        const x1 = from.x + nx * from.size;
        const y1 = from.y + ny * from.size;
        const x2 = to.x - nx * (to.size + 8);
        const y2 = to.y - ny * (to.size + 8);

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x1);
        line.setAttribute('y2', y1);
        line.setAttribute('stroke', 'rgba(59, 130, 246, 0.5)');
        line.setAttribute('stroke-width', '2');
        line.setAttribute('marker-end', 'url(#citationArrow)');
        line.style.opacity = '0';
        line.dataset.x2 = x2;
        line.dataset.y2 = y2;
        parent.appendChild(line);
        return line;
    }

    async animateNodeAppear(node, duration) {
        node.group.style.transformOrigin = `${node.x}px ${node.y}px`;
        node.group.style.transform = 'scale(0)';
        node.group.style.opacity = '1';

        await this.animate((t) => {
            const scale = this.easeOutElastic(Math.min(t * 1.2, 1));
            node.group.style.transform = `scale(${scale})`;
        }, duration);
    }

    async animateEdge(edge, duration) {
        const x1 = parseFloat(edge.getAttribute('x1'));
        const y1 = parseFloat(edge.getAttribute('y1'));
        const x2 = parseFloat(edge.dataset.x2);
        const y2 = parseFloat(edge.dataset.y2);

        edge.style.opacity = '1';
        await this.animate((t) => {
            edge.setAttribute('x2', this.lerp(x1, x2, t));
            edge.setAttribute('y2', this.lerp(y1, y2, t));
        }, duration);
    }

    async animateNetworkPulse(nodes) {
        await this.animate((t) => {
            const pulse = Math.sin(t * Math.PI * 3) * 0.15;
            nodes.forEach(node => {
                node.group.style.transform = `scale(${1 + pulse})`;
            });
        }, 1000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}


// ============================================================================
// 4. Hybrid Retrieval Scoring Animation
// ============================================================================

class HybridRetrievalAnimation extends ManimAnimation {
    constructor(container, options = {}) {
        super(container, { width: 700, height: 400, ...options });
    }

    init() {
        const svg = this.createSVG(700, 400);
        this.container.appendChild(svg);
        this.svg = svg;

        // Defs
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'barGlow');
        filter.innerHTML = '<feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>';
        defs.appendChild(filter);
        svg.appendChild(defs);

        // Title
        this.title = this.createText(350, 30, 'Hybrid Retrieval Score Computation', 16, 'bold');
        this.title.style.opacity = '0';

        const replayBtn = this.addReplayButton(svg);

        this.observeVisibility(async () => {
            await this.playAnimation();
            this.showReplayButton(replayBtn);
        });
    }

    createText(x, y, text, size, weight, fill = '#e2e8f0') {
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('fill', fill);
        t.setAttribute('font-size', size);
        t.setAttribute('font-weight', weight);
        t.setAttribute('font-family', 'system-ui');
        t.textContent = text;
        this.svg.appendChild(t);
        return t;
    }

    async playAnimation() {
        // Fade in title
        await this.animate((t) => {
            this.title.style.opacity = t;
        }, 500);

        // Component scores (will animate these bars)
        const components = [
            { label: 'Embedding Similarity', value: 0.85, weight: 0.40, color: '#3b82f6' },
            { label: 'Citation Proximity', value: 0.72, weight: 0.35, color: '#10b981' },
            { label: 'BM25 Text Match', value: 0.68, weight: 0.25, color: '#f59e0b' }
        ];

        const barWidth = 400;
        const barHeight = 35;
        const startY = 80;
        const spacing = 70;

        // Create bar groups
        const barGroups = [];

        for (let i = 0; i < components.length; i++) {
            const c = components[i];
            const y = startY + i * spacing;

            const group = this.createBarGroup(100, y, barWidth, barHeight, c);
            barGroups.push({ group, component: c, y });
        }

        // Animate each bar appearing and filling
        for (let i = 0; i < barGroups.length; i++) {
            const bg = barGroups[i];
            await this.animateBarIn(bg.group, bg.component.value, barWidth, 600);
            await this.delay(300);
        }

        // Show formula
        const formula = this.createText(350, 300, 'Score = 0.40 * S_embed + 0.35 * S_cite + 0.25 * S_text', 13, 'normal', '#10b981');
        formula.setAttribute('font-family', 'monospace');
        formula.style.opacity = '0';
        await this.animate((t) => { formula.style.opacity = t; }, 500);

        await this.delay(500);

        // Animate weighted combination
        const finalScore = components.reduce((sum, c) => sum + c.value * c.weight, 0);

        // Create final score bar
        const finalY = 340;
        const finalGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        const finalLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        finalLabel.setAttribute('x', 100);
        finalLabel.setAttribute('y', finalY + 20);
        finalLabel.setAttribute('fill', '#e2e8f0');
        finalLabel.setAttribute('font-size', '13');
        finalLabel.setAttribute('font-weight', 'bold');
        finalLabel.textContent = 'Final Score:';

        const finalBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        finalBg.setAttribute('x', 200);
        finalBg.setAttribute('y', finalY);
        finalBg.setAttribute('width', barWidth);
        finalBg.setAttribute('height', barHeight);
        finalBg.setAttribute('rx', '6');
        finalBg.setAttribute('fill', 'rgba(255,255,255,0.1)');

        // Gradient for final bar
        const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
        gradient.setAttribute('id', 'finalGradient');
        gradient.innerHTML = `
            <stop offset="0%" stop-color="#3b82f6"/>
            <stop offset="40%" stop-color="#10b981"/>
            <stop offset="100%" stop-color="#f59e0b"/>
        `;
        this.svg.querySelector('defs').appendChild(gradient);

        const finalFill = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        finalFill.setAttribute('x', 200);
        finalFill.setAttribute('y', finalY);
        finalFill.setAttribute('width', '0');
        finalFill.setAttribute('height', barHeight);
        finalFill.setAttribute('rx', '6');
        finalFill.setAttribute('fill', 'url(#finalGradient)');
        finalFill.setAttribute('filter', 'url(#barGlow)');

        const finalValue = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        finalValue.setAttribute('x', 620);
        finalValue.setAttribute('y', finalY + 23);
        finalValue.setAttribute('fill', '#10b981');
        finalValue.setAttribute('font-size', '16');
        finalValue.setAttribute('font-weight', 'bold');
        finalValue.textContent = '0.00';

        finalGroup.appendChild(finalLabel);
        finalGroup.appendChild(finalBg);
        finalGroup.appendChild(finalFill);
        finalGroup.appendChild(finalValue);
        finalGroup.style.opacity = '0';
        this.svg.appendChild(finalGroup);

        await this.animate((t) => { finalGroup.style.opacity = t; }, 300);

        // Animate final bar filling
        await this.animate((t) => {
            const width = t * finalScore * barWidth;
            finalFill.setAttribute('width', width);
            finalValue.textContent = (t * finalScore).toFixed(2);
        }, 1000);

        // Pulse final score
        await this.animate((t) => {
            const scale = 1 + Math.sin(t * Math.PI * 2) * 0.05;
            finalValue.style.transform = `scale(${scale})`;
            finalValue.style.transformOrigin = 'center';
        }, 600);
    }

    createBarGroup(x, y, width, height, component) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.style.opacity = '0';

        // Label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', x);
        label.setAttribute('y', y - 8);
        label.setAttribute('fill', '#e2e8f0');
        label.setAttribute('font-size', '12');
        label.textContent = component.label;

        // Weight badge
        const weightBadge = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        weightBadge.setAttribute('x', x + 180);
        weightBadge.setAttribute('y', y - 8);
        weightBadge.setAttribute('fill', component.color);
        weightBadge.setAttribute('font-size', '11');
        weightBadge.setAttribute('font-weight', 'bold');
        weightBadge.textContent = `(weight: ${component.weight})`;

        // Background bar
        const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bg.setAttribute('x', x);
        bg.setAttribute('y', y);
        bg.setAttribute('width', width);
        bg.setAttribute('height', height);
        bg.setAttribute('rx', '6');
        bg.setAttribute('fill', 'rgba(255,255,255,0.1)');

        // Fill bar
        const fill = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        fill.setAttribute('x', x);
        fill.setAttribute('y', y);
        fill.setAttribute('width', '0');
        fill.setAttribute('height', height);
        fill.setAttribute('rx', '6');
        fill.setAttribute('fill', component.color);
        fill.setAttribute('filter', 'url(#barGlow)');
        fill.setAttribute('class', 'fill-bar');

        // Value text
        const value = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        value.setAttribute('x', x + width + 15);
        value.setAttribute('y', y + height/2 + 5);
        value.setAttribute('fill', component.color);
        value.setAttribute('font-size', '14');
        value.setAttribute('font-weight', 'bold');
        value.setAttribute('class', 'value-text');
        value.textContent = '0.00';

        group.appendChild(label);
        group.appendChild(weightBadge);
        group.appendChild(bg);
        group.appendChild(fill);
        group.appendChild(value);
        this.svg.appendChild(group);

        return group;
    }

    async animateBarIn(group, targetValue, maxWidth, duration) {
        const fill = group.querySelector('.fill-bar');
        const value = group.querySelector('.value-text');

        group.style.opacity = '1';
        group.style.transform = 'translateX(-20px)';

        await this.animate((t) => {
            group.style.transform = `translateX(${-20 * (1-t)}px)`;
            group.style.opacity = t;
        }, 200);

        await this.animate((t) => {
            const width = t * targetValue * maxWidth;
            fill.setAttribute('width', width);
            value.textContent = (t * targetValue).toFixed(2);
        }, duration);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}


// ============================================================================
// 5. Embedding Space Visualization Animation
// ============================================================================

class EmbeddingSpaceAnimation extends ManimAnimation {
    constructor(container, options = {}) {
        super(container, { width: 700, height: 450, ...options });
    }

    init() {
        const svg = this.createSVG(700, 450);
        this.container.appendChild(svg);
        this.svg = svg;

        // Defs
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        const filter = document.createElementNS('http://www.w3.org/2000/svg', 'filter');
        filter.setAttribute('id', 'pointGlow');
        filter.innerHTML = '<feGaussianBlur stdDeviation="2" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>';
        defs.appendChild(filter);

        svg.appendChild(defs);

        // Title
        this.title = this.createText(350, 30, '128-Dimensional Embedding Space (2D Projection)', 16, 'bold');
        this.title.style.opacity = '0';

        const replayBtn = this.addReplayButton(svg);

        this.observeVisibility(async () => {
            await this.playAnimation();
            this.showReplayButton(replayBtn);
        });
    }

    createText(x, y, text, size, weight, fill = '#e2e8f0') {
        const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', 'middle');
        t.setAttribute('fill', fill);
        t.setAttribute('font-size', size);
        t.setAttribute('font-weight', weight);
        t.setAttribute('font-family', 'system-ui');
        t.textContent = text;
        this.svg.appendChild(t);
        return t;
    }

    async playAnimation() {
        // Fade in title
        await this.animate((t) => {
            this.title.style.opacity = t;
        }, 500);

        // Draw axes
        await this.drawAxes();

        // Generate random points in clusters
        const clusters = [
            { cx: 250, cy: 200, color: '#10b981', label: 'Petitioner Wins', count: 15 },
            { cx: 450, cy: 280, color: '#ef4444', label: 'Respondent Wins', count: 12 }
        ];

        const allPoints = [];

        // Create cluster points
        for (const cluster of clusters) {
            for (let i = 0; i < cluster.count; i++) {
                const angle = Math.random() * Math.PI * 2;
                const radius = 30 + Math.random() * 60;
                const x = cluster.cx + Math.cos(angle) * radius;
                const y = cluster.cy + Math.sin(angle) * radius;
                allPoints.push({ x, y, color: cluster.color, cluster: cluster.label });
            }
        }

        // Animate points appearing
        const pointElements = [];
        for (const p of allPoints) {
            const point = this.createPoint(p.x, p.y, p.color);
            pointElements.push({ el: point, data: p });
        }

        // Animate points in batches
        const batchSize = 5;
        for (let i = 0; i < pointElements.length; i += batchSize) {
            const batch = pointElements.slice(i, i + batchSize);
            await Promise.all(batch.map(p => this.animatePointIn(p.el)));
            await this.delay(100);
        }

        // Add legend
        await this.addLegend(clusters);

        // Add query case
        await this.delay(500);
        const queryPoint = this.createPoint(350, 240, '#f59e0b', 12, true);
        queryPoint.style.opacity = '0';

        const queryLabel = this.createText(350, 280, 'Query Case', 11, 'normal', '#f59e0b');
        queryLabel.style.opacity = '0';

        await this.animate((t) => {
            queryPoint.style.opacity = t;
            queryLabel.style.opacity = t;
            const scale = this.easeOutElastic(t);
            queryPoint.style.transform = `scale(${scale})`;
            queryPoint.style.transformOrigin = '350px 240px';
        }, 800);

        // Draw similarity lines to nearest points
        await this.delay(300);
        const nearestPoints = this.findNearest(350, 240, allPoints, 5);

        for (const np of nearestPoints) {
            const line = this.createSimilarityLine(350, 240, np.x, np.y);
            await this.animateLine(line, 300);
        }

        // Show similarity score
        const scoreText = this.createText(350, 400, 'Top-5 precedents retrieved by embedding similarity', 12, 'normal', '#10b981');
        scoreText.style.opacity = '0';
        await this.animate((t) => { scoreText.style.opacity = t; }, 500);
    }

    async drawAxes() {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');

        // X axis
        const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        xAxis.setAttribute('x1', '100');
        xAxis.setAttribute('y1', '350');
        xAxis.setAttribute('x2', '600');
        xAxis.setAttribute('y2', '350');
        xAxis.setAttribute('stroke', 'rgba(148, 163, 184, 0.3)');
        xAxis.setAttribute('stroke-width', '1');

        // Y axis
        const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        yAxis.setAttribute('x1', '100');
        yAxis.setAttribute('y1', '60');
        yAxis.setAttribute('x2', '100');
        yAxis.setAttribute('y2', '350');
        yAxis.setAttribute('stroke', 'rgba(148, 163, 184, 0.3)');
        yAxis.setAttribute('stroke-width', '1');

        // Labels
        const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        xLabel.setAttribute('x', '350');
        xLabel.setAttribute('y', '380');
        xLabel.setAttribute('text-anchor', 'middle');
        xLabel.setAttribute('fill', '#64748b');
        xLabel.setAttribute('font-size', '11');
        xLabel.textContent = 'Principal Component 1';

        const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        yLabel.setAttribute('x', '60');
        yLabel.setAttribute('y', '200');
        yLabel.setAttribute('text-anchor', 'middle');
        yLabel.setAttribute('fill', '#64748b');
        yLabel.setAttribute('font-size', '11');
        yLabel.setAttribute('transform', 'rotate(-90, 60, 200)');
        yLabel.textContent = 'Principal Component 2';

        group.appendChild(xAxis);
        group.appendChild(yAxis);
        group.appendChild(xLabel);
        group.appendChild(yLabel);
        group.style.opacity = '0';
        this.svg.appendChild(group);

        await this.animate((t) => { group.style.opacity = t; }, 500);
    }

    createPoint(x, y, color, r = 6, special = false) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x);
        circle.setAttribute('cy', y);
        circle.setAttribute('r', r);
        circle.setAttribute('fill', color);
        circle.setAttribute('filter', 'url(#pointGlow)');
        if (special) {
            circle.setAttribute('stroke', 'white');
            circle.setAttribute('stroke-width', '2');
        }
        circle.style.opacity = '0';
        this.svg.appendChild(circle);
        return circle;
    }

    async animatePointIn(point) {
        await this.animate((t) => {
            point.style.opacity = t;
        }, 200);
    }

    async addLegend(clusters) {
        const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        legendGroup.style.opacity = '0';

        let y = 70;
        for (const cluster of clusters) {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', '570');
            circle.setAttribute('cy', y);
            circle.setAttribute('r', '6');
            circle.setAttribute('fill', cluster.color);

            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', '585');
            text.setAttribute('y', y + 4);
            text.setAttribute('fill', '#94a3b8');
            text.setAttribute('font-size', '11');
            text.textContent = cluster.label;

            legendGroup.appendChild(circle);
            legendGroup.appendChild(text);
            y += 25;
        }

        this.svg.appendChild(legendGroup);
        await this.animate((t) => { legendGroup.style.opacity = t; }, 400);
    }

    findNearest(x, y, points, k) {
        return points
            .map(p => ({ ...p, dist: Math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2) }))
            .sort((a, b) => a.dist - b.dist)
            .slice(0, k);
    }

    createSimilarityLine(x1, y1, x2, y2) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x1);
        line.setAttribute('y2', y1);
        line.setAttribute('stroke', 'rgba(245, 158, 11, 0.5)');
        line.setAttribute('stroke-width', '1');
        line.setAttribute('stroke-dasharray', '4,2');
        line.dataset.x2 = x2;
        line.dataset.y2 = y2;
        this.svg.insertBefore(line, this.svg.querySelector('circle'));
        return line;
    }

    async animateLine(line, duration) {
        const x1 = parseFloat(line.getAttribute('x1'));
        const y1 = parseFloat(line.getAttribute('y1'));
        const x2 = parseFloat(line.dataset.x2);
        const y2 = parseFloat(line.dataset.y2);

        await this.animate((t) => {
            line.setAttribute('x2', this.lerp(x1, x2, t));
            line.setAttribute('y2', this.lerp(y1, y2, t));
        }, duration);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}


// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize animations when their containers exist

    // 1. Pipeline Animation (index.html - System Architecture)
    const pipelineContainer = document.getElementById('pipeline-animation');
    if (pipelineContainer) {
        new PipelineAnimation(pipelineContainer).init();
    }

    // 2. GraphSAGE Animation (methodology.html)
    const graphsageContainer = document.getElementById('graphsage-animation');
    if (graphsageContainer) {
        new GraphSAGEAnimation(graphsageContainer).init();
    }

    // 3. Citation Network Animation (data.html)
    const citationContainer = document.getElementById('citation-network-animation');
    if (citationContainer) {
        new CitationNetworkAnimation(citationContainer).init();
    }

    // 4. Hybrid Retrieval Animation (methodology.html)
    const retrievalContainer = document.getElementById('hybrid-retrieval-animation');
    if (retrievalContainer) {
        new HybridRetrievalAnimation(retrievalContainer).init();
    }

    // 5. Embedding Space Animation (methodology.html)
    const embeddingContainer = document.getElementById('embedding-space-animation');
    if (embeddingContainer) {
        new EmbeddingSpaceAnimation(embeddingContainer).init();
    }
});

// Export for manual initialization
window.ManimAnimations = {
    PipelineAnimation,
    GraphSAGEAnimation,
    CitationNetworkAnimation,
    HybridRetrievalAnimation,
    EmbeddingSpaceAnimation
};
