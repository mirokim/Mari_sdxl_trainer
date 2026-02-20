'use client';
import { useEffect, useState, useCallback } from 'react';
import Header from '@/components/layout/Header';
import PageContainer from '@/components/layout/PageContainer';
import { datasetApi } from '@/lib/api';
import { DatasetInfo, DatasetImage } from '@/lib/types';

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [selected, setSelected] = useState<string>('');
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [newName, setNewName] = useState('');
  const [editingCaption, setEditingCaption] = useState<string | null>(null);
  const [captionText, setCaptionText] = useState('');

  const loadDatasets = useCallback(async () => {
    try {
      const res = await datasetApi.list();
      setDatasets(res.datasets);
    } catch {}
  }, []);

  const loadImages = useCallback(async (name: string) => {
    try {
      const res = await datasetApi.getImages(name);
      setImages(res.images);
      setStats(res.stats);
    } catch {}
  }, []);

  useEffect(() => { loadDatasets(); }, [loadDatasets]);
  useEffect(() => { if (selected) loadImages(selected); }, [selected, loadImages]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await datasetApi.create(newName.trim());
    setNewName('');
    loadDatasets();
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length || !selected) return;
    await datasetApi.uploadImages(selected, files);
    loadImages(selected);
  };

  const handleSaveCaption = async (imagePath: string) => {
    await datasetApi.updateCaption(selected, imagePath, captionText);
    setEditingCaption(null);
    loadImages(selected);
  };

  return (
    <PageContainer>
      <Header title="데이터셋" description="학습 이미지 관리 및 캡션 편집" />

      <div className="flex gap-6">
        {/* 좌측: 데이터셋 목록 */}
        <div className="w-64 flex-shrink-0">
          <div className="notion-section">
            <h2 className="notion-section-title">데이터셋 목록</h2>
            <div className="space-y-1">
              {datasets.map((ds) => (
                <button
                  key={ds.name}
                  onClick={() => setSelected(ds.name)}
                  className={`w-full text-left px-3 py-2 rounded-md text-notion-small transition-colors
                    ${selected === ds.name
                      ? 'bg-notion-accent text-white'
                      : 'hover:bg-notion-hover text-notion-text'
                    }`}
                >
                  <div className="font-medium">{ds.name}</div>
                  <div className="text-[11px] opacity-75">{ds.image_count}개 이미지</div>
                </button>
              ))}
            </div>
          </div>

          {/* 새 데이터셋 생성 */}
          <div className="notion-section">
            <div className="flex gap-2">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="새 데이터셋 이름"
                className="notion-input text-notion-small flex-1"
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
              />
              <button onClick={handleCreate} className="notion-btn-primary">+</button>
            </div>
          </div>
        </div>

        {/* 우측: 이미지 갤러리 & 캡션 */}
        <div className="flex-1 min-w-0">
          {selected ? (
            <>
              {/* 업로드 */}
              <div className="flex items-center gap-4 mb-4">
                <label className="notion-btn-secondary cursor-pointer">
                  이미지 업로드
                  <input type="file" multiple accept="image/*" className="hidden" onChange={handleUpload} />
                </label>
                {stats && (
                  <span className="text-notion-small text-notion-text-secondary">
                    {stats.total}개 이미지 · {stats.captioned}개 캡션 완료
                  </span>
                )}
              </div>

              {/* 이미지 그리드 */}
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {images.map((img) => (
                  <div key={img.path} className="notion-card p-2 group">
                    <div className="aspect-square bg-notion-hover rounded overflow-hidden mb-2">
                      <img
                        src={`/files/datasets/${img.relative_path}`}
                        alt={img.filename}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <p className="text-[11px] text-notion-text font-medium truncate">
                      {img.filename}
                    </p>
                    <p className="text-[11px] text-notion-text-secondary">
                      {img.width}x{img.height}
                    </p>

                    {/* 캡션 편집 */}
                    {editingCaption === img.path ? (
                      <div className="mt-2">
                        <textarea
                          value={captionText}
                          onChange={(e) => setCaptionText(e.target.value)}
                          className="notion-input text-[11px] h-16 resize-none"
                        />
                        <div className="flex gap-1 mt-1">
                          <button
                            onClick={() => handleSaveCaption(img.path)}
                            className="notion-btn-primary text-[11px] py-0.5"
                          >
                            저장
                          </button>
                          <button
                            onClick={() => setEditingCaption(null)}
                            className="notion-btn-ghost text-[11px] py-0.5"
                          >
                            취소
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={() => {
                          setEditingCaption(img.path);
                          setCaptionText(img.caption);
                        }}
                        className="mt-1 text-[11px] text-notion-text-secondary
                                 hover:text-notion-accent transition-colors text-left w-full"
                      >
                        {img.caption
                          ? img.caption.substring(0, 60) + (img.caption.length > 60 ? '...' : '')
                          : '캡션 추가...'}
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-64 text-notion-text-secondary">
              좌측에서 데이터셋을 선택하세요
            </div>
          )}
        </div>
      </div>
    </PageContainer>
  );
}
