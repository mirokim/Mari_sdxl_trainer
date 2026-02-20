interface PageContainerProps {
  children: React.ReactNode;
}

export default function PageContainer({ children }: PageContainerProps) {
  return (
    <div className="notion-page animate-fade-in">
      {children}
    </div>
  );
}
