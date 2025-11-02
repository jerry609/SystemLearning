package main

import (
	"fmt"
	"strings"
	"time"
)

// FileSystemNode 文件系统节点接口
type FileSystemNode interface {
	GetName() string
	GetSize() int64
	GetPath() string
	GetPermissions() string
	SetPermissions(perms string)
	GetModifiedTime() time.Time
	Display(indent string)
	Copy() FileSystemNode
}

// File 文件
type File struct {
	name         string
	size         int64
	permissions  string
	modifiedTime time.Time
	parent       *Directory
}

func NewFile(name string, size int64) *File {
	return &File{
		name:         name,
		size:         size,
		permissions:  "rw-",
		modifiedTime: time.Now(),
	}
}

func (f *File) GetName() string {
	return f.name
}

func (f *File) GetSize() int64 {
	return f.size
}

func (f *File) GetPath() string {
	if f.parent == nil {
		return f.name
	}
	parentPath := f.parent.GetPath()
	if parentPath == "" {
		return f.name
	}
	return parentPath + "/" + f.name
}

func (f *File) GetPermissions() string {
	return f.permissions
}

func (f *File) SetPermissions(perms string) {
	f.permissions = perms
}

func (f *File) GetModifiedTime() time.Time {
	return f.modifiedTime
}

func (f *File) Display(indent string) {
	fmt.Printf("%s📄 %s (%d KB) [%s]\n", indent, f.name, f.size, f.permissions)
}

func (f *File) Copy() FileSystemNode {
	return &File{
		name:         f.name,
		size:         f.size,
		permissions:  f.permissions,
		modifiedTime: time.Now(),
	}
}

// Directory 目录
type Directory struct {
	name         string
	permissions  string
	modifiedTime time.Time
	children     []FileSystemNode
	parent       *Directory
}

func NewDirectory(name string) *Directory {
	return &Directory{
		name:         name,
		permissions:  "rwx",
		modifiedTime: time.Now(),
		children:     make([]FileSystemNode, 0),
	}
}

func (d *Directory) GetName() string {
	return d.name
}

func (d *Directory) GetSize() int64 {
	var total int64
	for _, child := range d.children {
		total += child.GetSize()
	}
	return total
}

func (d *Directory) GetPath() string {
	if d.parent == nil {
		return ""
	}
	parentPath := d.parent.GetPath()
	if parentPath == "" {
		return d.name
	}
	return parentPath + "/" + d.name
}

func (d *Directory) GetPermissions() string {
	return d.permissions
}

func (d *Directory) SetPermissions(perms string) {
	d.permissions = perms
}

func (d *Directory) GetModifiedTime() time.Time {
	return d.modifiedTime
}

func (d *Directory) Display(indent string) {
	fmt.Printf("%s📁 %s/ (%d KB) [%s]\n", indent, d.name, d.GetSize(), d.permissions)
	for _, child := range d.children {
		child.Display(indent + "  ")
	}
}

func (d *Directory) Copy() FileSystemNode {
	newDir := &Directory{
		name:         d.name,
		permissions:  d.permissions,
		modifiedTime: time.Now(),
		children:     make([]FileSystemNode, 0),
	}
	
	for _, child := range d.children {
		childCopy := child.Copy()
		if file, ok := childCopy.(*File); ok {
			file.parent = newDir
		} else if dir, ok := childCopy.(*Directory); ok {
			dir.parent = newDir
		}
		newDir.children = append(newDir.children, childCopy)
	}
	
	return newDir
}

// Add 添加子节点
func (d *Directory) Add(node FileSystemNode) {
	if file, ok := node.(*File); ok {
		file.parent = d
	} else if dir, ok := node.(*Directory); ok {
		dir.parent = d
	}
	d.children = append(d.children, node)
	d.modifiedTime = time.Now()
}

// Remove 移除子节点
func (d *Directory) Remove(node FileSystemNode) {
	for i, child := range d.children {
		if child == node {
			d.children = append(d.children[:i], d.children[i+1:]...)
			d.modifiedTime = time.Now()
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
		
		if dir, ok := child.(*Directory); ok {
			if found := dir.Find(name); found != nil {
				return found
			}
		}
	}
	
	return nil
}

// Search 搜索文件（支持通配符）
func (d *Directory) Search(pattern string) []FileSystemNode {
	results := make([]FileSystemNode, 0)
	
	for _, child := range d.children {
		if strings.Contains(child.GetName(), pattern) {
			results = append(results, child)
		}
		
		if dir, ok := child.(*Directory); ok {
			results = append(results, dir.Search(pattern)...)
		}
	}
	
	return results
}

// CountFiles 统计文件总数
func (d *Directory) CountFiles() int {
	count := 0
	
	for _, child := range d.children {
		if _, ok := child.(*File); ok {
			count++
		} else if dir, ok := child.(*Directory); ok {
			count += dir.CountFiles()
		}
	}
	
	return count
}

// ListAllFiles 列出所有文件路径
func (d *Directory) ListAllFiles() []string {
	files := make([]string, 0)
	
	for _, child := range d.children {
		if file, ok := child.(*File); ok {
			files = append(files, file.GetPath())
		} else if dir, ok := child.(*Directory); ok {
			files = append(files, dir.ListAllFiles()...)
		}
	}
	
	return files
}

// Move 移动节点到另一个目录
func (d *Directory) Move(node FileSystemNode, target *Directory) error {
	// 从当前目录移除
	d.Remove(node)
	
	// 添加到目标目录
	target.Add(node)
	
	return nil
}
