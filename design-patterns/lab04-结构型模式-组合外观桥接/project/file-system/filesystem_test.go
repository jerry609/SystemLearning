package main

import (
	"testing"
)

func TestFileCreation(t *testing.T) {
	file := NewFile("test.txt", 100)
	
	if file.GetName() != "test.txt" {
		t.Errorf("Expected name 'test.txt', got '%s'", file.GetName())
	}
	
	if file.GetSize() != 100 {
		t.Errorf("Expected size 100, got %d", file.GetSize())
	}
	
	if file.GetPermissions() != "rw-" {
		t.Errorf("Expected permissions 'rw-', got '%s'", file.GetPermissions())
	}
}

func TestDirectoryCreation(t *testing.T) {
	dir := NewDirectory("testdir")
	
	if dir.GetName() != "testdir" {
		t.Errorf("Expected name 'testdir', got '%s'", dir.GetName())
	}
	
	if dir.GetSize() != 0 {
		t.Errorf("Expected size 0, got %d", dir.GetSize())
	}
	
	if dir.GetPermissions() != "rwx" {
		t.Errorf("Expected permissions 'rwx', got '%s'", dir.GetPermissions())
	}
}

func TestAddFile(t *testing.T) {
	dir := NewDirectory("testdir")
	file := NewFile("test.txt", 100)
	
	dir.Add(file)
	
	if len(dir.GetChildren()) != 1 {
		t.Errorf("Expected 1 child, got %d", len(dir.GetChildren()))
	}
	
	if dir.GetSize() != 100 {
		t.Errorf("Expected size 100, got %d", dir.GetSize())
	}
}

func TestRemoveFile(t *testing.T) {
	dir := NewDirectory("testdir")
	file := NewFile("test.txt", 100)
	
	dir.Add(file)
	dir.Remove(file)
	
	if len(dir.GetChildren()) != 0 {
		t.Errorf("Expected 0 children, got %d", len(dir.GetChildren()))
	}
	
	if dir.GetSize() != 0 {
		t.Errorf("Expected size 0, got %d", dir.GetSize())
	}
}

func TestNestedDirectories(t *testing.T) {
	root := NewDirectory("root")
	dir1 := NewDirectory("dir1")
	dir2 := NewDirectory("dir2")
	file1 := NewFile("file1.txt", 100)
	file2 := NewFile("file2.txt", 200)
	
	root.Add(dir1)
	dir1.Add(dir2)
	dir2.Add(file1)
	dir2.Add(file2)
	
	expectedSize := int64(300)
	if root.GetSize() != expectedSize {
		t.Errorf("Expected size %d, got %d", expectedSize, root.GetSize())
	}
}

func TestFindFile(t *testing.T) {
	root := NewDirectory("root")
	dir := NewDirectory("dir")
	file := NewFile("test.txt", 100)
	
	root.Add(dir)
	dir.Add(file)
	
	found := root.Find("test.txt")
	if found == nil {
		t.Error("Expected to find file, got nil")
	}
	
	if found != nil && found.GetName() != "test.txt" {
		t.Errorf("Expected name 'test.txt', got '%s'", found.GetName())
	}
}

func TestFindDirectory(t *testing.T) {
	root := NewDirectory("root")
	dir := NewDirectory("testdir")
	
	root.Add(dir)
	
	found := root.Find("testdir")
	if found == nil {
		t.Error("Expected to find directory, got nil")
	}
	
	if found != nil && found.GetName() != "testdir" {
		t.Errorf("Expected name 'testdir', got '%s'", found.GetName())
	}
}

func TestSearch(t *testing.T) {
	root := NewDirectory("root")
	file1 := NewFile("test1.txt", 100)
	file2 := NewFile("test2.txt", 200)
	file3 := NewFile("other.pdf", 300)
	
	root.Add(file1)
	root.Add(file2)
	root.Add(file3)
	
	results := root.Search(".txt")
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

func TestCountFiles(t *testing.T) {
	root := NewDirectory("root")
	dir1 := NewDirectory("dir1")
	dir2 := NewDirectory("dir2")
	file1 := NewFile("file1.txt", 100)
	file2 := NewFile("file2.txt", 200)
	file3 := NewFile("file3.txt", 300)
	
	root.Add(dir1)
	root.Add(dir2)
	dir1.Add(file1)
	dir1.Add(file2)
	dir2.Add(file3)
	
	count := root.CountFiles()
	if count != 3 {
		t.Errorf("Expected 3 files, got %d", count)
	}
}

func TestCopyFile(t *testing.T) {
	file := NewFile("test.txt", 100)
	file.SetPermissions("r--")
	
	copy := file.Copy().(*File)
	
	if copy.GetName() != file.GetName() {
		t.Errorf("Expected name '%s', got '%s'", file.GetName(), copy.GetName())
	}
	
	if copy.GetSize() != file.GetSize() {
		t.Errorf("Expected size %d, got %d", file.GetSize(), copy.GetSize())
	}
	
	if copy.GetPermissions() != file.GetPermissions() {
		t.Errorf("Expected permissions '%s', got '%s'", file.GetPermissions(), copy.GetPermissions())
	}
}

func TestCopyDirectory(t *testing.T) {
	dir := NewDirectory("testdir")
	file1 := NewFile("file1.txt", 100)
	file2 := NewFile("file2.txt", 200)
	
	dir.Add(file1)
	dir.Add(file2)
	
	copy := dir.Copy().(*Directory)
	
	if copy.GetName() != dir.GetName() {
		t.Errorf("Expected name '%s', got '%s'", dir.GetName(), copy.GetName())
	}
	
	if len(copy.GetChildren()) != len(dir.GetChildren()) {
		t.Errorf("Expected %d children, got %d", len(dir.GetChildren()), len(copy.GetChildren()))
	}
	
	if copy.GetSize() != dir.GetSize() {
		t.Errorf("Expected size %d, got %d", dir.GetSize(), copy.GetSize())
	}
}

func TestMove(t *testing.T) {
	source := NewDirectory("source")
	target := NewDirectory("target")
	file := NewFile("test.txt", 100)
	
	source.Add(file)
	
	if len(source.GetChildren()) != 1 {
		t.Errorf("Expected 1 child in source, got %d", len(source.GetChildren()))
	}
	
	source.Move(file, target)
	
	if len(source.GetChildren()) != 0 {
		t.Errorf("Expected 0 children in source, got %d", len(source.GetChildren()))
	}
	
	if len(target.GetChildren()) != 1 {
		t.Errorf("Expected 1 child in target, got %d", len(target.GetChildren()))
	}
}

func TestGetPath(t *testing.T) {
	root := NewDirectory("root")
	dir := NewDirectory("dir")
	file := NewFile("test.txt", 100)
	
	root.Add(dir)
	dir.Add(file)
	
	expectedPath := "dir/test.txt"
	if file.GetPath() != expectedPath {
		t.Errorf("Expected path '%s', got '%s'", expectedPath, file.GetPath())
	}
}

func TestSetPermissions(t *testing.T) {
	file := NewFile("test.txt", 100)
	
	file.SetPermissions("r--")
	if file.GetPermissions() != "r--" {
		t.Errorf("Expected permissions 'r--', got '%s'", file.GetPermissions())
	}
	
	dir := NewDirectory("testdir")
	dir.SetPermissions("r-x")
	if dir.GetPermissions() != "r-x" {
		t.Errorf("Expected permissions 'r-x', got '%s'", dir.GetPermissions())
	}
}
