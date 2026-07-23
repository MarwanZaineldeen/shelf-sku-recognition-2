import { Link } from "react-router-dom";
import { Compass, ScanSearch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/common/states";

export function NotFoundPage() {
  return (
    <div className="mx-auto flex min-h-[60dvh] max-w-lg items-center px-6">
      <EmptyState
        icon={Compass}
        title="Page not found"
        description="That route does not exist in the audit suite. Head back to the shelf workspace to start a new audit."
        action={
          <Button asChild>
            <Link to="/audit">
              <ScanSearch aria-hidden />
              Go to Shelf Audit
            </Link>
          </Button>
        }
      />
    </div>
  );
}

export default NotFoundPage;
