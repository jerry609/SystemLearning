package main

import (
	"fmt"
)

// 组合模式示例：文件系统
// 本示例展示了如何使用组合模式表示文件系统的树形结构

// FileSystemNode 文件系统节点接口（Component）
type FileSystemNode interface {
	GetName() string
	GetSize() int64
	Display(indent string)
}

// File 文件（Leaf）
type File struct {
	name string
	size int64
}

func NewFile(name string, size int64) *File {
	return &File{
		name: name,
		size: size,
	}
}

func (f *File) GetName() string {
	return f.name
}

func (f *File) GetSize() int64 {
	return f.size
}

func (f *File) Display(indent string) {
	fmt.Printf("%s📄 %s (%d KB)\n", indent, f.name, f.size)
}

// Directory 目录（Composite）
type Directory struct {
	name     string
	children []FileSystemNode
}

func NewDirectory(name string) *Directory {
	return &Directory{
		name:     name,
		children: make([]FileSystemNode, 0),
	}
}

func (d *Directory) GetName() string {
	return d.name
}

func (d *Directory) GetSize() int64 {
	var totalSize int64
	for _, child := range d.children {
		totalSize += child.GetSize()
	}
	return totalSize
}

func (d *Directory) Display(indent string) {
	fmt.Printf("%s📁 %s/ (%d KB)\n", indent, d.name, d.GetSize())
	for _, child := range d.children {
		child.Display(indent + "  ")
	}
}

// Add 添加子节点
func (d *Directory) Add(node FileSystemNode) {
	d.children = append(d.children, node)
}

// Remove 移除子节点
func (d *Directory) Remove(node FileSystemNode) {
	for i, child := range d.children {
		if child == node {
			d.children = append(d.children[:i], d.children[i+1:]...)
			break
		}
	}
}

// GetChildren 获取所有子节点
func (d *Directory) GetChildren() []FileSystemNode {
	return d.children
}

// Find 查找文件或目录
func (d *Directory) Find(name string) FileSystemNode {
	if d.name == name {
		return d
	}
	
	for _, child := range d.children {
		if child.GetName() == name {
			return child
		}
		
		// 如果子节点是目录，递归查找
		if dir, ok := child.(*Directory); ok {
			if found := dir.Find(name); found != nil {
				return found
			}
		}
	}
	
	return nil
}

// ListFiles 列出所有文件（递归）
func (d *Directory) ListFiles() []string {
	files := make([]string, 0)
	
	for _, child := range d.children {
		if file, ok := child.(*File); ok {
			files = append(files, file.GetName())
		} else if dir, ok := child.(*Directory); ok {
			subFiles := dir.ListFiles()
			for _, f := range subFiles {
				files = append(files, dir.GetName()+"/"+f)
			}
		}
	}
	
	return files
}

func main() {
	fmt.Println("=== 组合模式示例：文件系统 ===\n")

	// 创建根目录
	root := NewDirectory("root")

	// 创建子目录
	documents := NewDirectory("documents")
	pictures := NewDirectory("pictures")
	videos := NewDirectory("videos")

	// 创建文件
	readme := NewFile("README.md", 5)
	report := NewFile("report.pdf", 1024)
	photo1 := NewFile("vacation.jpg", 2048)
	photo2 := NewFile("family.jpg", 1536)
	movie := NewFile("movie.mp4", 10240)

	// 构建树形结构
	root.Add(documents)
	root.Add(pictures)
	root.Add(videos)

	documents.Add(readme)
	documents.Add(report)

	pictures.Add(photo1)
	pictures.Add(photo2)

	videos.Add(movie)

	// 显示文件系统结构
	fmt.Println("文件系统结构：")
	root.Display("")

	// 计算总大小
	fmt.Printf("\n总大小：%d KB\n", root.GetSize())

	// 查找文件
	fmt.Println("\n查找文件 'report.pdf'：")
	found := root.Find("report.pdf")
	if found != nil {
		fmt.Printf("找到：%s (大小：%d KB)\n", found.GetName(), found.GetSize())
	}

	// 列出所有文件
	fmt.Println("\n所有文件列表：")
	files := root.ListFiles()
	for _, file := range files {
		fmt.Printf("  - %s\n", file)
	}

	// 移除目录
	fmt.Println("\n移除 videos 目录后：")
	root.Remove(videos)
	root.Display("")
	fmt.Printf("总大小：%d KB\n", root.GetSize())

	fmt.Println("\n=== 示例结束 ===")
}

// 输出示例：
// === 组合模式示例：文件系统 ===
//
// 文件系统结构：
// 📁 root/ (14853 KB)
//   📁 documents/ (1029 KB)
//     📄 README.md (5 KB)
//     📄 report.pdf (1024 KB)
//   📁 pictures/ (3584 KB)
//     📄 vacation.jpg (2048 KB)
//     📄 family.jpg (1536 KB)
//   📁 videos/ (10240 KB)
//     📄 movie.mp4 (10240 KB)
//
// 总大小：14853 KB
//
// 查找文件 'report.pdf'：
// 找到：report.pdf (大小：1024 KB)
//
// 所有文件列表：
//   - documents/README.md
//   - documents/report.pdf
//   - pictures/vacation.jpg
//   - pictures/family.jpg
//   - videos/movie.mp4
//
// 移除 videos 目录后：
// 📁 root/ (4613 KB)
//   📁 documents/ (1029 KB)
//     📄 README.md (5 KB)
//     📄 report.pdf (1024 KB)
//   📁 pictures/ (3584 KB)
//     📄 vacation.jpg (2048 KB)
//     📄 family.jpg (1536 KB)
// 总大小：4613 KB
//
// === 示例结束 ===
