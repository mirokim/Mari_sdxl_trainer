'use client';
import { useEffect, useState } from 'react';
import { systemApi } from '@/lib/api';
import { GpuInfo } from '@/lib/types';

interface HeaderProps {
  title: string;
  description?: string;
}

export default function Header({ title, description }: HeaderProps) {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);

  useEffect(() => {
    systemApi.getGpuInfo().then(setGpu).catch(() => {});
  }, []);

  return (
    <div className="mb-8">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-notion-title text-notion-text">{title}</h1>
          {description && (
            <p className="mt-1 text-notion-body text-notion-text-secondary">
              {description}
            </p>
          )}
        </div>

        {gpu?.available && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-notion-sidebar
                        rounded-md border border-notion-border text-notion-small">
            <span className="w-2 h-2 rounded-full bg-notion-success" />
            <span className="text-notion-text-secondary">
              {gpu.name} · {gpu.free_gb}GB free
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
