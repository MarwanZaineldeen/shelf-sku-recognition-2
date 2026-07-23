import * as React from "react";
import { ErrorState } from "./states";

interface Props {
  children: React.ReactNode;
  /** Remounting key — change it to clear the error (e.g. the route pathname). */
  resetKey?: string;
}

interface State {
  error: Error | null;
}

/**
 * Catches render-time crashes so one broken panel cannot blank the whole app.
 * Wrap routes and any widget doing risky work (canvas, charts, third-party).
 */
export class ErrorBoundary extends React.Component<Props, State> {
  override state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  override componentDidUpdate(prevProps: Props) {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("Unhandled UI error:", error, info.componentStack);
  }

  private readonly reset = () => this.setState({ error: null });

  override render() {
    if (this.state.error) {
      return (
        <div className="p-6">
          <ErrorState
            title="This view failed to render"
            error={this.state.error}
            onRetry={this.reset}
          />
        </div>
      );
    }
    return this.props.children;
  }
}
