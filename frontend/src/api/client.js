/**
 * API client for the Deep Research Agent.
 *
 * Connects directly to the FastAPI backend with comprehensive error handling,
 * response validation, and automatic API base fallback.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Custom API error with HTTP status and detail from backend.
 */
export class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (response.status === 204) {
      return null;
    }

    const contentType = response.headers.get('content-type');
    let data = null;
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      const detail =
        (data && typeof data === 'object' && (data.detail || data.message)) ||
        (typeof data === 'string' && data) ||
        response.statusText ||
        'Request failed';
      throw new ApiError(detail, response.status, data);
    }

    return data;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    // Network failure or abort
    throw new ApiError(
      error.message || 'Network connection error. Is the backend server running?',
      0,
      null
    );
  }
}

export const api = {
  /**
   * Health check to test backend connection.
   */
  async healthCheck() {
    return request('/api/health');
  },

  /**
   * List recent research sessions.
   */
  async listSessions(limit = 30) {
    return request(`/api/research?limit=${encodeURIComponent(limit)}`);
  },

  /**
   * Get full details of a specific session.
   */
  async getSession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}`);
  },

  /**
   * Fetch session, node events, and review state in one workbench request.
   */
  async getWorkbenchRefresh(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/workbench`);
  },

  /**
   * Create a new research session.
   */
  async createSession({
    question,
    require_initial_plan_approval = true,
    require_initial_research_approval = true,
    require_adaptive_research_approval = true,
    require_final_report_approval = true,
    max_research_rounds = 3,
    max_total_research_questions = 10,
  }) {
    return request('/api/research', {
      method: 'POST',
      body: JSON.stringify({
        question,
        require_initial_plan_approval,
        require_initial_research_approval,
        require_adaptive_research_approval,
        require_final_report_approval,
        max_research_rounds: Number(max_research_rounds),
        max_total_research_questions: Number(max_total_research_questions),
      }),
    });
  },

  /**
   * Get pending review for a session waiting for approval.
   */
  async getPendingReview(sessionId) {
    try {
      return await request(`/api/research/${encodeURIComponent(sessionId)}/review`);
    } catch (err) {
      // 409 means not waiting for review
      if (err.status === 409) {
        return null;
      }
      throw err;
    }
  },

  /**
   * Submit human review decision.
   */
  async submitReview(sessionId, decision) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/review`, {
      method: 'POST',
      body: JSON.stringify(decision),
    });
  },

  /**
   * Retry a failed or stopped session.
   */
  async retrySession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/retry`, {
      method: 'POST',
    });
  },

  /**
   * Cancel queued research or request a safe worker stop.
   */
  async cancelSession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/cancel`, {
      method: 'POST',
    });
  },

  /**
   * Pause research at a safe workflow checkpoint.
   */
  async pauseSession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/pause`, {
      method: 'POST',
    });
  },

  /**
   * Resume a paused research job from its latest checkpoint.
   */
  async resumeSession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/resume`, {
      method: 'POST',
    });
  },

  /**
   * Delete a research session and all cascaded data.
   */
  async deleteSession(sessionId) {
    return request(`/api/research/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    });
  },

  /**
   * Get observability events for a session.
   */
  async getSessionEvents(sessionId, limit = 150) {
    return request(`/api/research/${encodeURIComponent(sessionId)}/events?limit=${encodeURIComponent(limit)}`);
  },

  /**
   * Get global observability events across all sessions.
   */
  async getGlobalEvents(limit = 100) {
    return request(`/api/observability/events?limit=${encodeURIComponent(limit)}`);
  },

  /**
   * Get server run metrics snapshot.
   */
  async getMetrics() {
    return request('/api/observability/metrics');
  },
};
