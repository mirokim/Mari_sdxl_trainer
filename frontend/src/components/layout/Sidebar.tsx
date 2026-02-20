'use client';
import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { clsx } from 'clsx';

const NAV_ITEMS = [
  { href: '/', label: '대시보드', icon: '◈' },
  { href: '/dataset', label: '데이터셋', icon: '◫' },
  { href: '/captioning', label: '캡셔닝', icon: '◧' },
  { href: '/training', label: '학습', icon: '▶' },
  { href: '/output', label: '결과물', icon: '◆' },
  { href: '/preview', label: '미리보기', icon: '◩' },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={clsx(
        'fixed left-0 top-0 h-screen bg-notion-sidebar border-r border-notion-border',
        'flex flex-col transition-all duration-200 z-50',
        collapsed ? 'w-[48px]' : 'w-[240px]',
      )}
    >
      {/* 헤더 */}
      <div className="flex items-center justify-between px-3 py-3 min-h-[48px]">
        {!collapsed && (
          <span className="text-notion-small font-semibold text-notion-text truncate">
            Mari SDXL Trainer
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded hover:bg-notion-hover text-notion-text-secondary
                     transition-colors duration-100 flex-shrink-0"
        >
          {collapsed ? '→' : '←'}
        </button>
      </div>

      {/* 네비게이션 */}
      <nav className="flex-1 px-1 py-1 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                'flex items-center gap-2.5 px-2.5 py-1.5 rounded-md',
                'text-notion-small transition-all duration-100',
                isActive
                  ? 'bg-notion-hover text-notion-text font-medium'
                  : 'text-notion-text-secondary hover:bg-notion-hover hover:text-notion-text',
              )}
            >
              <span className="text-base flex-shrink-0 w-5 text-center">
                {item.icon}
              </span>
              {!collapsed && <span className="truncate">{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* 하단 정보 */}
      {!collapsed && (
        <div className="px-3 py-3 border-t border-notion-border">
          <p className="text-[11px] text-notion-text-secondary">
            v1.0.0 · ComfyUI 호환
          </p>
        </div>
      )}
    </aside>
  );
}
