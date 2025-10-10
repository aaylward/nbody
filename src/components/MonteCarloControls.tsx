import { useState } from 'react';

export function MonteCarloControls() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  return (
    <div className={`control-panel ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="control-header">
        <h3>Monte Carlo Transport</h3>
        <button
          className="toggle-button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? 'Expand controls' : 'Collapse controls'}
        >
          {isCollapsed ? '▶' : '◀'}
        </button>
      </div>

      {!isCollapsed && (
        <div className="demo-notice">Monte Carlo controls coming soon!</div>
      )}
    </div>
  );
}
