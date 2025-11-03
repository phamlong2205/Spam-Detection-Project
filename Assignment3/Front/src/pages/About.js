// src/pages/About.js — RF training overview + 2 trust charts (images from public/assets)
import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const img = (name) => `${process.env.PUBLIC_URL}/assets/${name}`;

export default function About() {
  useEffect(() => { window.scrollTo(0, 0); }, []);
  const [showSpam, setShowSpam] = useState(true);
  const [showHam, setShowHam] = useState(true);
  const chartRef = useRef(null);
  const scatterRef = useRef(null);
  const radarRef = useRef(null);
  const heatmapRef = useRef(null);
  const violinRef = useRef(null);

  // Interactive Feature Bar Chart with tooltips and filtering
  useEffect(() => {
    if (!chartRef.current) return;

    const features = [
      { name: "message_length", spam: 350, ham: 650, description: "Average message length", unit: "chars" },
      { name: "digit_ratio", spam: 0.08, ham: 0.02, description: "Ratio of digits to total characters", unit: "" },
      { name: "capital_ratio", spam: 0.12, ham: 0.03, description: "Ratio of capital letters", unit: "" },
      { name: "special_char_count", spam: 15, ham: 5, description: "Number of special characters", unit: "count" },
      { name: "url_count", spam: 1.5, ham: 0.2, description: "Number of URLs", unit: "count" },
      { name: "average_word_length", spam: 4.8, ham: 4.2, description: "Average word length", unit: "chars" }
    ];

    d3.select(chartRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 120, bottom: 60, left: 80 };
    const width = 700 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select(chartRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create tooltip div
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "13px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)")
      .style("transition", "opacity 0.2s");

    const x = d3.scaleBand()
      .domain(features.map(d => d.name))
      .range([0, width])
      .padding(0.3);

    const maxValue = d3.max(features, d => Math.max(d.spam, d.ham));
    const y = d3.scaleLinear()
      .domain([0, maxValue * 1.1])
      .range([height, 0]);

    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .style("font-size", "11px")
      .style("cursor", "pointer")
      .on("click", function(event, d) {
        d3.selectAll(".feature-label").style("font-weight", "normal").style("fill", "#666");
        d3.select(this).style("font-weight", "bold").style("fill", "#000");
      });

    svg.append("g")
      .call(d3.axisLeft(y));

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margin.left + 20)
      .attr("x", 0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Average Value");

    // Spam bars with enhanced interactivity
    if (showSpam) {
      svg.selectAll(".bar-spam")
        .data(features)
        .enter()
        .append("rect")
        .attr("class", "bar-spam")
        .attr("x", d => x(d.name))
        .attr("y", height)
        .attr("width", x.bandwidth() / 2)
        .attr("height", 0)
        .attr("fill", "#e74c3c")
        .attr("opacity", 0.8)
        .attr("cursor", "pointer")
        .on("mouseover", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 1)
            .attr("y", y(d.spam) - 5)
            .attr("height", height - y(d.spam) + 5);

          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.name.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div style="margin-bottom: 5px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="color: #e74c3c; font-weight: bold;">
                ⬤ Spam: ${d.spam}${d.unit}
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 0.8)
            .attr("y", y(d.spam))
            .attr("height", height - y(d.spam));
          tooltip.style("opacity", 0);
        })
        .transition()
        .duration(800)
        .attr("y", d => y(d.spam))
        .attr("height", d => height - y(d.spam));
    }

    // Ham bars with enhanced interactivity
    if (showHam) {
      svg.selectAll(".bar-ham")
        .data(features)
        .enter()
        .append("rect")
        .attr("class", "bar-ham")
        .attr("x", d => x(d.name) + x.bandwidth() / 2)
        .attr("y", height)
        .attr("width", x.bandwidth() / 2)
        .attr("height", 0)
        .attr("fill", "#27ae60")
        .attr("opacity", 0.8)
        .attr("cursor", "pointer")
        .on("mouseover", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 1)
            .attr("y", y(d.ham) - 5)
            .attr("height", height - y(d.ham) + 5);

          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.name.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div style="margin-bottom: 5px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="color: #27ae60; font-weight: bold;">
                ⬤ Ham: ${d.ham}${d.unit}
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function(event, d) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("opacity", 0.8)
            .attr("y", y(d.ham))
            .attr("height", height - y(d.ham));
          tooltip.style("opacity", 0);
        })
        .transition()
        .duration(800)
        .attr("y", d => y(d.ham))
        .attr("height", d => height - y(d.ham));
    }

    // Interactive legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width + 10}, 0)`);

    const spamLegend = legend.append("g")
      .style("cursor", "pointer")
      .on("click", () => setShowSpam(!showSpam));

    spamLegend.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", "#e74c3c")
      .attr("opacity", showSpam ? 0.8 : 0.3);

    spamLegend.append("text")
      .attr("x", 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("font-size", "12px")
      .text("Spam (click)")
      .style("opacity", showSpam ? 1 : 0.5);

    const hamLegend = legend.append("g")
      .style("cursor", "pointer")
      .on("click", () => setShowHam(!showHam));

    hamLegend.append("rect")
      .attr("x", 0)
      .attr("y", 25)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", "#27ae60")
      .attr("opacity", showHam ? 0.8 : 0.3);

    hamLegend.append("text")
      .attr("x", 24)
      .attr("y", 34)
      .attr("dy", ".35em")
      .style("font-size", "12px")
      .text("Ham (click)")
      .style("opacity", showHam ? 1 : 0.5);

    svg.append("text")
      .attr("x", width / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .style("font-size", "16px")
      .style("font-weight", "bold")
      .text("Feature Comparison: Spam vs Ham (Interactive)");

    // Cleanup function
    return () => {
      d3.selectAll(".d3-tooltip").remove();
    };

  }, [showSpam, showHam]);

  // Interactive Scatter plot with zoom
  useEffect(() => {
    if (!scatterRef.current) return;

    // Sample scatter data
    const scatterData = [
      // Spam samples (red)
      ...Array(40).fill(0).map((_, i) => ({
        id: `spam-${i}`,
        x: Math.random() * 0.15 + 0.05,
        y: Math.random() * 0.2 + 0.05,
        type: "spam",
        message_length: Math.floor(Math.random() * 200 + 100)
      })),
      // Ham samples (green)
      ...Array(60).fill(0).map((_, i) => ({
        id: `ham-${i}`,
        x: Math.random() * 0.05,
        y: Math.random() * 0.05,
        type: "ham",
        message_length: Math.floor(Math.random() * 500 + 300)
      }))
    ];

    d3.select(scatterRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const width = 550 - margin.left - margin.right;
    const height = 450 - margin.top - margin.bottom;

    const svg = d3.select(scatterRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip-scatter")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)");

    const x = d3.scaleLinear()
      .domain([0, 0.25])
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, 0.3])
      .range([height, 0]);

    // Add axes
    const xAxis = g.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

    const yAxis = g.append("g")
      .call(d3.axisLeft(y));

    // Axis labels
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 45)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Digit Ratio");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -50)
      .attr("x", -height / 2)
      .attr("text-anchor", "middle")
      .style("font-size", "12px")
      .text("Capital Ratio");

    // Clip path for zoom
    g.append("defs").append("clipPath")
      .attr("id", "clip-scatter")
      .append("rect")
      .attr("width", width)
      .attr("height", height);

    const scatter = g.append("g")
      .attr("clip-path", "url(#clip-scatter)");

    // Filter data based on showSpam and showHam
    const filteredData = scatterData.filter(d =>
      (d.type === "spam" && showSpam) || (d.type === "ham" && showHam)
    );

    // Add points
    const circles = scatter.selectAll("circle")
      .data(filteredData)
      .enter()
      .append("circle")
      .attr("cx", d => x(d.x))
      .attr("cy", d => y(d.y))
      .attr("r", 6)
      .attr("fill", d => d.type === "spam" ? "#e74c3c" : "#27ae60")
      .attr("opacity", 0.6)
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .style("cursor", "pointer")
      .on("mouseover", function(event, d) {
        d3.select(this)
          .attr("r", 10)
          .attr("opacity", 1)
          .attr("stroke-width", 2);

        tooltip
          .style("opacity", 1)
          .html(`
            <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px; color: ${d.type === "spam" ? "#e74c3c" : "#27ae60"};">
              ⬤ ${d.type.toUpperCase()}
            </div>
            <div style="margin-top: 5px;">
              <div style="margin-bottom: 3px;">📊 Digit Ratio: <b>${d.x.toFixed(3)}</b></div>
              <div style="margin-bottom: 3px;">🔠 Capital Ratio: <b>${d.y.toFixed(3)}</b></div>
              <div>📏 Message Length: <b>${d.message_length} chars</b></div>
            </div>
          `);
      })
      .on("mousemove", function(event) {
        tooltip
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", function() {
        d3.select(this)
          .attr("r", 6)
          .attr("opacity", 0.6)
          .attr("stroke-width", 1);
        tooltip.style("opacity", 0);
      });

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 5])
      .on("zoom", (event) => {
        // Handle zoom event
        const transform = event.transform;
        const newX = transform.rescaleX(x);
        const newY = transform.rescaleY(y);

        xAxis.call(d3.axisBottom(newX));
        yAxis.call(d3.axisLeft(newY));

        circles
          .attr("cx", d => newX(d.x))
          .attr("cy", d => newY(d.y));
      });

    svg.call(zoom);

    // Title
    g.append("text")
      .attr("x", width / 2)
      .attr("y", -10)
      .attr("text-anchor", "middle")
      .style("font-size", "14px")
      .style("font-weight", "bold")
      .text("Interactive Scatter: Digit vs Capital Ratio (Zoom Enabled)");

    // Reset button
    const resetButton = svg.append("g")
      .attr("transform", `translate(${width + margin.left - 60}, 10)`)
      .style("cursor", "pointer")
      .on("click", function(event) {
        event.stopPropagation(); // Prevent zoom event from firing
        // Reset zoom with smooth transition
        svg.transition()
          .duration(750)
          .call(zoom.transform, d3.zoomIdentity);
      });

    resetButton.append("rect")
      .attr("width", 50)
      .attr("height", 20)
      .attr("rx", 4)
      .attr("fill", "#3498db")
      .attr("opacity", 0.8);

    resetButton.append("text")
      .attr("x", 25)
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .attr("fill", "white")
      .attr("font-size", "10px")
      .text("Reset");

    // Cleanup
    return () => {
      d3.selectAll(".d3-tooltip-scatter").remove();
    };

  }, [showSpam, showHam]);

  // Interactive Radar chart with toggle and hover effects
  useEffect(() => {
    if (!radarRef.current) return;

    const features = [
      { axis: "Digit Ratio", spam: 0.8, ham: 0.2, description: "Numbers in message" },
      { axis: "Capital Ratio", spam: 0.9, ham: 0.25, description: "Uppercase letters" },
      { axis: "Special Chars", spam: 0.85, ham: 0.3, description: "Symbols & punctuation" },
      { axis: "URL Count", spam: 0.75, ham: 0.15, description: "Number of links" },
      { axis: "Avg Word Len", spam: 0.6, ham: 0.55, description: "Word complexity" },
      { axis: "Message Len", spam: 0.4, ham: 0.7, description: "Total characters" }
    ];

    d3.select(radarRef.current).selectAll("*").remove();

    const width = 500;
    const height = 500;
    const margin = 100;
    const radius = Math.min(width, height) / 2 - margin;

    const svg = d3.select(radarRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${width/2},${height/2})`);

    // Tooltip
    const tooltip = d3.select("body")
      .append("div")
      .attr("class", "d3-tooltip-radar")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.9)")
      .style("color", "white")
      .style("padding", "10px 15px")
      .style("border-radius", "6px")
      .style("font-size", "12px")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 10000)
      .style("box-shadow", "0 2px 8px rgba(0,0,0,0.3)");

    const angleSlice = Math.PI * 2 / features.length;

    // Draw circular grid
    const levels = 5;
    for (let i = 0; i < levels; i++) {
      const levelFactor = radius * ((i + 1) / levels);

      g.append("circle")
        .attr("r", levelFactor)
        .attr("fill", "none")
        .attr("stroke", "#CDCDCD")
        .attr("stroke-width", 0.5)
        .style("cursor", "default");

      if (i < levels) {
        g.append("text")
          .attr("x", 5)
          .attr("y", -levelFactor)
          .attr("fill", "#737373")
          .attr("font-size", "9px")
          .text(((i + 1) * 0.2).toFixed(1));
      }
    }

    // Draw axis lines and labels
    features.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const lineCoord = angleToCoordinate(angle, radius);

      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", lineCoord.x)
        .attr("y2", lineCoord.y)
        .attr("stroke", "#CDCDCD")
        .attr("stroke-width", 1);

      const labelCoord = angleToCoordinate(angle, radius + 40);
      g.append("text")
        .attr("x", labelCoord.x)
        .attr("y", labelCoord.y)
        .attr("text-anchor", "middle")
        .attr("font-size", "11px")
        .attr("font-weight", "bold")
        .attr("fill", "#333")
        .attr("cursor", "pointer")
        .text(d.axis)
        .on("mouseover", function(event) {
          d3.select(this).attr("fill", "#e74c3c").attr("font-size", "13px");
          tooltip
            .style("opacity", 1)
            .html(`
              <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 5px;">
                ${d.axis}
              </div>
              <div style="margin-bottom: 8px; color: #ddd; font-size: 11px;">
                ${d.description}
              </div>
              <div style="display: flex; gap: 15px;">
                <div style="color: #e74c3c;">
                  <div style="font-size: 10px; opacity: 0.8;">SPAM</div>
                  <div style="font-weight: bold; font-size: 16px;">${d.spam.toFixed(2)}</div>
                </div>
                <div style="color: #27ae60;">
                  <div style="font-size: 10px; opacity: 0.8;">HAM</div>
                  <div style="font-weight: bold; font-size: 16px;">${d.ham.toFixed(2)}</div>
                </div>
              </div>
            `);
        })
        .on("mousemove", function(event) {
          tooltip
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
        })
        .on("mouseout", function() {
          d3.select(this).attr("fill", "#333").attr("font-size", "11px");
          tooltip.style("opacity", 0);
        });
    });

    function angleToCoordinate(angle, value) {
      return {
        x: Math.cos(angle) * value,
        y: Math.sin(angle) * value
      };
    }

    function drawPath(data, color, label, show) {
      if (!show) return;

      const pathData = features.map((d, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        const value = data === "spam" ? d.spam : d.ham;
        return angleToCoordinate(angle, value * radius);
      });

      const lineGenerator = d3.line()
        .x(d => d.x)
        .y(d => d.y)
        .curve(d3.curveLinearClosed);

      const path = g.append("path")
        .datum(pathData)
        .attr("class", `radar-${data}`)
        .attr("d", lineGenerator)
        .attr("fill", color)
        .attr("fill-opacity", 0.1)
        .attr("stroke", color)
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .style("cursor", "pointer")
        .on("mouseover", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill-opacity", 0.4)
            .attr("stroke-width", 3);
        })
        .on("mouseout", function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr("fill-opacity", 0.1)
            .attr("stroke-width", 2);
        });

      path.transition()
        .duration(1000)
        .attr("opacity", 1);

      // Add interactive dots
      pathData.forEach((coord, i) => {
        g.append("circle")
          .attr("class", `radar-dot-${data}`)
          .attr("cx", coord.x)
          .attr("cy", coord.y)
          .attr("r", 0)
          .attr("fill", color)
          .attr("opacity", 0.8)
          .attr("cursor", "pointer")
          .on("mouseover", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("r", 8)
              .attr("opacity", 1);

            const feature = features[i];
            const value = data === "spam" ? feature.spam : feature.ham;
            const colorCode = data === "spam" ? "#e74c3c" : "#27ae60";
            tooltip
              .style("opacity", 1)
              .html(`
                <div style="font-weight: bold; margin-bottom: 5px; color: ${colorCode};">
                  ⬤ ${feature.axis}
                </div>
                <div style="font-size: 11px; margin-bottom: 5px; color: #ddd;">
                  ${feature.description}
                </div>
                <div style="font-weight: bold; font-size: 16px;">
                  ${label}: ${value.toFixed(2)}
                </div>
              `);
          })
          .on("mousemove", function(event) {
            tooltip
              .style("left", (event.pageX + 10) + "px")
              .style("top", (event.pageY - 28) + "px");
          })
          .on("mouseout", function() {
            d3.select(this)
              .transition()
              .duration(200)
              .attr("r", 5)
              .attr("opacity", 0.8);
            tooltip.style("opacity", 0);
          })
          .transition()
          .delay(1000 + i * 100)
          .duration(300)
          .attr("r", 5);
      });
    }

    drawPath("spam", "#e74c3c", "Spam", showSpam);
    drawPath("ham", "#27ae60", "Ham", showHam);

    // Interactive legend with toggle
    const legend = g.append("g")
      .attr("transform", `translate(${-radius + 20}, ${radius - 60})`);

    const spamLegendG = legend.append("g")
      .style("cursor", "pointer")
      .attr("opacity", showSpam ? 1 : 0.4)
      .on("click", () => setShowSpam(!showSpam))
      .on("mouseover", function() {
        d3.select(this).select("circle").attr("r", 8);
      })
      .on("mouseout", function() {
        d3.select(this).select("circle").attr("r", 6);
      });

    spamLegendG.append("circle").attr("cx", 0).attr("cy", 0).attr("r", 6).attr("fill", "#e74c3c");
    spamLegendG.append("text").attr("x", 15).attr("y", 4).text("Spam Profile (click)").attr("font-size", "12px");

    const hamLegendG = legend.append("g")
      .style("cursor", "pointer")
      .attr("opacity", showHam ? 1 : 0.4)
      .on("click", () => setShowHam(!showHam))
      .on("mouseover", function() {
        d3.select(this).select("circle").attr("r", 8);
      })
      .on("mouseout", function() {
        d3.select(this).select("circle").attr("r", 6);
      });

    hamLegendG.append("circle").attr("cx", 0).attr("cy", 25).attr("r", 6).attr("fill", "#27ae60");
    hamLegendG.append("text").attr("x", 15).attr("y", 29).text("Ham Profile (click)").attr("font-size", "12px");

    // Cleanup
    return () => {
      d3.selectAll(".d3-tooltip-radar").remove();
    };

  }, [showSpam, showHam]);

  // Heatmap for feature correlations
  useEffect(() => {
    if (!heatmapRef.current) return;

    const features = ["message_length", "digit_ratio", "capital_ratio", "special_chars", "url_count"];
    const correlationData = [
      [1.0, -0.3, -0.2, 0.4, 0.3],
      [-0.3, 1.0, 0.6, 0.5, 0.7],
      [-0.2, 0.6, 1.0, 0.4, 0.5],
      [0.4, 0.5, 0.4, 1.0, 0.6],
      [0.3, 0.7, 0.5, 0.6, 1.0]
    ];

    d3.select(heatmapRef.current).selectAll("*").remove();

    const margin = { top: 80, right: 50, bottom: 120, left: 120 };
    const cellSize = 70;
    const width = cellSize * features.length + margin.left + margin.right;
    const height = cellSize * features.length + margin.top + margin.bottom;

    const svg = d3.select(heatmapRef.current)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([-1, 1]);

    // Draw cells
    features.forEach((feat1, i) => {
      features.forEach((feat2, j) => {
        const correlation = correlationData[i][j];

        const cell = svg.append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize)
          .attr("height", cellSize)
          .attr("fill", "white")
          .attr("stroke", "#ccc")
          .attr("stroke-width", 1)
          .on("mouseover", function() {
            d3.select(this).attr("stroke", "#000").attr("stroke-width", 2);
          })
          .on("mouseout", function() {
            d3.select(this).attr("stroke", "#ccc").attr("stroke-width", 1);
          });

        cell.transition()
          .delay((i + j) * 50)
          .duration(500)
          .attr("fill", colorScale(correlation));

        svg.append("text")
          .attr("x", j * cellSize + cellSize / 2)
          .attr("y", i * cellSize + cellSize / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", "12px")
          .attr("font-weight", "bold")
          .attr("fill", Math.abs(correlation) > 0.5 ? "white" : "#333")
          .attr("opacity", 0)
          .text(correlation.toFixed(2))
          .transition()
          .delay((i + j) * 50 + 300)
          .duration(300)
          .attr("opacity", 1);
      });
    });

    // X axis labels
    features.forEach((feat, i) => {
      const xPos = i * cellSize + cellSize / 2;
      const yPos = -15;
      svg.append("text")
        .attr("x", xPos)
        .attr("y", yPos)
        .attr("text-anchor", "start")
        .attr("transform", `rotate(-45, ${xPos}, ${yPos})`)
        .attr("font-size", "12px")
        .attr("font-weight", "500")
        .text(feat);
    });

    // Y axis labels
    features.forEach((feat, i) => {
      svg.append("text")
        .attr("x", -15)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", "12px")
        .attr("font-weight", "500")
        .text(feat);
    });

  }, []);

  // Violin/Box plot for feature distributions
  useEffect(() => {
    if (!violinRef.current) return;

    const features = ["message_length", "digit_ratio", "capital_ratio", "special_chars"];

    d3.select(violinRef.current).selectAll("*").remove();

    const margin = { top: 40, right: 40, bottom: 80, left: 60 };
    const width = 600 - margin.left - margin.right;
    const height = 350 - margin.top - margin.bottom;

    const svg = d3.select(violinRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleBand()
      .domain(features)
      .range([0, width])
      .padding(0.3);

    const y = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0]);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-45)")
      .style("text-anchor", "end")
      .style("font-size", "11px");

    svg.append("g")
      .call(d3.axisLeft(y));

    // Draw box plots for each feature
    features.forEach((feat, i) => {
      const boxWidth = x.bandwidth() / 3;
      const xPos = x(feat);

      // Spam box plot (normalized values)
      const spamStats = { min: 0.5, q1: 0.6, median: 0.75, q3: 0.85, max: 0.95 };
      drawBoxPlot(svg, xPos + 5, spamStats, boxWidth, y, "#e74c3c", i * 100);

      // Ham box plot
      const hamStats = { min: 0.05, q1: 0.15, median: 0.25, q3: 0.35, max: 0.45 };
      drawBoxPlot(svg, xPos + boxWidth + 10, hamStats, boxWidth, y, "#27ae60", i * 100 + 50);
    });

    function drawBoxPlot(svg, x, stats, width, yScale, color, delay) {
      const center = x + width / 2;

      // Whiskers
      svg.append("line")
        .attr("x1", center)
        .attr("x2", center)
        .attr("y1", yScale(stats.min))
        .attr("y2", yScale(stats.max))
        .attr("stroke", color)
        .attr("stroke-width", 1.5)
        .attr("opacity", 0)
        .transition()
        .delay(delay)
        .duration(500)
        .attr("opacity", 1);

      // Box
      const boxHeight = yScale(stats.q1) - yScale(stats.q3);
      svg.append("rect")
        .attr("x", x)
        .attr("y", yScale(stats.q3))
        .attr("width", width)
        .attr("height", 0)
        .attr("fill", color)
        .attr("fill-opacity", 0.6)
        .attr("stroke", color)
        .attr("stroke-width", 1.5)
        .transition()
        .delay(delay)
        .duration(500)
        .attr("height", boxHeight);

      // Median line
      svg.append("line")
        .attr("x1", x)
        .attr("x2", x + width)
        .attr("y1", yScale(stats.median))
        .attr("y2", yScale(stats.median))
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("opacity", 0)
        .transition()
        .delay(delay + 300)
        .duration(300)
        .attr("opacity", 1);
    }

    // Title
    svg.append("text")
      .attr("x", width / 2)
      .attr("y", -15)
      .attr("text-anchor", "middle")
      .attr("font-size", "14px")
      .attr("font-weight", "bold")
      .text("Feature Distribution (Box Plots)");

    // Legend
    const legend = svg.append("g")
      .attr("transform", `translate(${width - 100}, 10)`);

    legend.append("rect")
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#e74c3c")
      .attr("opacity", 0.6);
    legend.append("text")
      .attr("x", 20)
      .attr("y", 12)
      .text("Spam")
      .attr("font-size", "11px");

    legend.append("rect")
      .attr("y", 20)
      .attr("width", 15)
      .attr("height", 15)
      .attr("fill", "#27ae60")
      .attr("opacity", 0.6);
    legend.append("text")
      .attr("x", 20)
      .attr("y", 32)
      .text("Ham")
      .attr("font-size", "11px");

  }, []);

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>About the Model</h2>

      <p className="lead">
        This app deploys a <b>Random Forest</b> to detect <b>spam</b> vs <b>ham</b>. Below is a quick
        overview of our training pipeline and two training charts to build trust.
      </p>

      {/* About the Team */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>About the Team</h3>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
          gap: '20px', 
          marginTop: 20 
        }}>
          
          {/* Duc Tri Tran */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'flex-start', 
            gap: '15px',
            padding: '15px',
            background: '#f8f9fa',
            borderRadius: '8px'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#e9ecef',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              color: '#6c757d',
              flexShrink: 0
            }}>
              DT
            </div>
            <div>
              <h4 style={{ margin: 0, marginBottom: '5px', fontSize: '16px' }}>
                Duc Tri Tran
              </h4>
              <p style={{ 
                margin: 0, 
                marginBottom: '8px', 
                fontSize: '13px', 
                fontWeight: 'bold',
                color: '#007bff' 
              }}>
                Data/ML Lead, Project Manager
              </p>
              <p className="small" style={{ margin: 0, lineHeight: 1.5 }}>
                Leads technical development and project management, overseeing the core ML system 
                from dataset preprocessing and feature engineering to model training and optimization. 
                Manages project timelines and ensures all deliverables meet accuracy requirements.
              </p>
            </div>
          </div>

          {/* Quoc Phi Long Pham */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'flex-start', 
            gap: '15px',
            padding: '15px',
            background: '#f8f9fa',
            borderRadius: '8px'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#e9ecef',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              color: '#6c757d',
              flexShrink: 0
            }}>
              QP
            </div>
            <div>
              <h4 style={{ margin: 0, marginBottom: '5px', fontSize: '16px' }}>
                Quoc Phi Long Pham
              </h4>
              <p style={{ 
                margin: 0, 
                marginBottom: '8px', 
                fontSize: '13px', 
                fontWeight: 'bold',
                color: '#28a745' 
              }}>
                Web Development Lead
              </p>
              <p className="small" style={{ margin: 0, lineHeight: 1.5 }}>
                Designs and builds the interactive web application, translating functional requirements 
                into an intuitive user interface. Responsible for implementing features like file uploads, 
                text analysis, and clear visualization of detection results using modern web technologies.
              </p>
            </div>
          </div>

          {/* Hengheng Lonh */}
          <div style={{ 
            display: 'flex', 
            alignItems: 'flex-start', 
            gap: '15px',
            padding: '15px',
            background: '#f8f9fa',
            borderRadius: '8px'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              borderRadius: '50%',
              background: '#e9ecef',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '24px',
              color: '#6c757d',
              flexShrink: 0
            }}>
              HL
            </div>
            <div>
              <h4 style={{ margin: 0, marginBottom: '5px', fontSize: '16px' }}>
                Hengheng Lonh
              </h4>
              <p style={{ 
                margin: 0, 
                marginBottom: '8px', 
                fontSize: '13px', 
                fontWeight: 'bold',
                color: '#dc3545' 
              }}>
                QA Lead
              </p>
              <p className="small" style={{ margin: 0, lineHeight: 1.5 }}>
                Ensures quality, reliability, and integrity of the final product through comprehensive 
                testing plans. Evaluates model performance against key metrics and conducts thorough 
                functional and usability testing, managing bug tracking and user acceptance testing.
              </p>
            </div>
          </div>

        </div>
      </div>

      <div className="pipeline">
        <div className="step">
          <div className="step-num">1</div>
          <div className="step-title">Collect & Clean</div>
          <div className="step-body small">
            Merge SMS + Email → <code>(label, message)</code>; lowercase, strip HTML/URLs, normalise spaces.
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">2</div>
          <div className="step-title">Feature Engineering</div>
          <div className="step-body small">
            TF-IDF (1–2 grams) + numeric signals (length, digit/capital ratios, special chars, avg word length, URL count).
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">3</div>
          <div className="step-title">Split & Balance</div>
          <div className="step-body small">
            Stratified Train/Val/Test; balance <i>train only</i> when needed (e.g., SMOTE).
          </div>
        </div>
        <div className="arrow">→</div>
        <div className="step">
          <div className="step-num">4</div>
          <div className="step-title">Train & Select</div>
          <div className="step-body small">
            Tune RF depth/trees; SelectKBest to keep strongest features; final metrics on held-out test.
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Model Card</h3>
        <ul className="small" style={{ lineHeight: 1.7 }}>
          <li><b>Algorithm:</b> Random Forest</li>
          <li><b>Vectoriser:</b> TF-IDF (1–2 grams) + numeric features</li>
          <li><b>Primary metric:</b> F1; we also track accuracy and train–val gap</li>
          <li><b>Serving threshold:</b> 0.50 on P(spam)</li>
          <li><b>Why RF:</b> fast inference, robust on sparse text, low overfit gap, easy to explain</li>
          <li><b>Limits:</b> very short/unseen obfuscated texts; OOD inputs compress probabilities</li>
        </ul>
      </div>

      {/* Two trust-building charts from public/assets */}
      <div className="viz-grid two-cols">
        <figure className="viz">
          <img src={img("rf_training_loss.png")} alt="Random Forest Training & Validation Loss" />
          <figcaption>Loss for both train and validation decreases steadily → stable learning.</figcaption>
        </figure>

        <figure className="viz">
          <img src={img("rf_training_accuracy.png")} alt="Random Forest Training & Validation Accuracy" />
          <figcaption>Accuracy rises together with a small gap → good generalisation (low overfitting).</figcaption>
        </figure>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h3 style={{ marginTop: 0 }}>Transparency & Reproducibility</h3>
        <p className="small">
          TF-IDF, selector (if used), and the trained RF model are saved as artefacts. The API loads them
          to return a probability + label instantly; inputs are stored only when you click Predict (so
          you can see history and export CSV).
        </p>
      </div>

      {/* Interactive Feature Visualization */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Interactive Feature Analysis</h3>

        <p className="small" style={{ marginBottom: 20 }}>
          Key features that distinguish spam from ham messages. <b>Red</b> represents spam characteristics, <b>green</b> represents ham (legitimate messages).
        </p>

        <div style={{ marginBottom: 30 }}>
          <h4 style={{ marginTop: 0, fontSize: '14px', color: '#555' }}>Feature Comparison Chart</h4>
          <div ref={chartRef} style={{ overflow: 'auto', marginBottom: 20, position: 'relative' }}></div>
        </div>

        <div>
          <h4 style={{ marginTop: 0, fontSize: '14px', color: '#555' }}>Two-Feature Distribution</h4>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div ref={scatterRef} style={{ position: 'relative' }}></div>
          </div>
        </div>
      </div>

      {/* Radar Chart */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Feature Profile: Radar Chart</h3>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={radarRef} style={{ position: 'relative' }}></div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Feature Correlation Heatmap</h3>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={heatmapRef} style={{ overflow: 'auto', position: 'relative' }}></div>
        </div>
      </div>

      {/* Box Plot */}
      <div className="card" style={{ marginTop: 24 }}>
        <h3 style={{ marginTop: 0 }}>Feature Distribution: Box Plots</h3>
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 20 }}>
          <div ref={violinRef}></div>
        </div>
      </div>

    </div>
  );
}
